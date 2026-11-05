import argparse
import json
import re
from pathlib import Path
 
import pandas as pd
 
ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "veri"
C = ROOT / "ciktilar"
C.mkdir(parents=True, exist_ok=True)
 
KONULAR = ["randevu", "fiyat", "süre", "personel", "iş_kalitesi", "yedek_parça", "temizlik", "diğer"]
DUYGULAR = ["Olumlu", "Olumsuz", "Karışık", "Alakasız"]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
 
SISTEM = f"""
Sen müşteri geri bildirimlerini etiketleyen bir analistsin. Metni duygu ve konu bakımından sınıflandır.
Yalnızca geçerli bir JSON nesnesi döndür; açıklama, Markdown veya kod bloğu ekleme.
Örnek: {{"duygu": "Olumlu", "konular": ["personel", "iş_kalitesi"]}}
Duygu etiketleri: {", ".join(DUYGULAR)}
Konu etiketleri: {", ".join(KONULAR)}
Kurallar:
1. Hem övgü hem şikâyet varsa duygu "Karışık" olmalıdır.
2. Servisle ilgisi olmayan metin "Alakasız", konular ["diğer"] olmalıdır.
3. Açıkça geçmeyen veya ima edilmeyen konuyu ekleme.
4. Birden fazla konu varsa uygun tüm konuları ekle.
5. İroni varsa gerçek niyete göre etiketle.
6. Yalnızca izin verilen etiketleri kullan.
7. "konular" mutlaka liste olmalıdır.
8. Kararsız kalırsan açık kanıtlara bağlı kal.
""".strip()
 
 
def parse_response(response_text):
    if response_text is None or not str(response_text).strip():
        raise ValueError("Model boş yanıt döndürdü.")
    text = str(response_text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Yanıtta JSON nesnesi bulunamadı: {text[:160]!r}")
    data = json.loads(text[start:end + 1])
    duygu, konular = data.get("duygu"), data.get("konular")
    if duygu not in DUYGULAR:
        raise ValueError(f"Geçersiz duygu etiketi: {duygu!r}")
    if not isinstance(konular, list) or not all(isinstance(k, str) for k in konular):
        raise ValueError("'konular' metinlerden oluşan bir liste olmalıdır.")
    konular = list(dict.fromkeys(k.strip() for k in konular if k.strip()))
    bad = [k for k in konular if k not in KONULAR]
    if bad:
        raise ValueError(f"Geçersiz konu etiketleri: {bad}")
    if not konular:
        raise ValueError("En az bir konu etiketi bulunmalıdır.")
    if duygu == "Alakasız":
        konular = ["diğer"]
    return {"duygu": duygu, "konular": konular}
 
 
def ornek():
    input_file = V / "geri_bildirimler.csv"
    if not input_file.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {input_file}")
    g = pd.read_csv(input_file)
    g = g.dropna(subset=["metin"]).copy()
    g["metin"] = g["metin"].astype(str).str.strip()
    g = g[g["metin"] != ""]
    sample = g.sample(n=min(200, len(g)), random_state=42).copy()
    sample.to_csv(C / "ornek_200.csv", index=False, encoding="utf-8-sig")
    gold = sample.sample(n=min(40, len(sample)), random_state=7)[["geri_bildirim_id", "metin"]].copy()
    gold["duygu"] = ""
    gold["konular"] = ""
    gold.to_csv(C / "altin_set.csv", index=False, encoding="utf-8-sig")
    print(f"Örnek: {len(sample)} | Altın set: {len(gold)}")
    print("altin_set.csv dosyasını model çıktısına bakmadan elle doldurun; çoklu konuları ; ile ayırın.")
 
 
def load_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Model yükleniyor: {model_name} (ilk kullanımda indirilir)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    print("Model hazır. Cihaz: CPU", flush=True)
    return tokenizer, model
 
 
def tek_metin_etiketle(tokenizer, model, metin, max_new_tokens=180):
    import torch
    messages = [
        {"role": "system", "content": SISTEM},
        {"role": "user", "content": f"Müşteri geri bildirimi:\n{metin}"},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    n_in = enc["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return parse_response(tokenizer.decode(out[0][n_in:], skip_special_tokens=True))
 
 
def etiketle(model_name=MODEL, limit=None):
    input_file = C / "ornek_200.csv"
    if not input_file.exists():
        raise FileNotFoundError("ornek_200.csv yok. Önce 'ornek' adımını çalıştırın.")
    sample = pd.read_csv(input_file)
    if limit is not None:
        sample = sample.head(limit).copy()
 
    out_file = C / "llm_etiketleri.csv"
    cache = {}
    if out_file.exists():
        eski = pd.read_csv(out_file, encoding="utf-8-sig")
        for r in eski.itertuples(index=False):
            if r.duygu != "HATA":
                cache[str(r.metin)] = (r.duygu, r.konular)
 
    benzersiz = [t for t in sample["metin"].astype(str).unique() if t not in cache]
    print(f"Toplam kayıt: {len(sample)} | Benzersiz metin: {sample['metin'].nunique()} | Etiketlenecek: {len(benzersiz)}", flush=True)
 
    tokenizer = model = None
    if benzersiz:
        tokenizer, model = load_model(model_name)
 
    hatalar = {}
    for n, metin in enumerate(benzersiz, 1):
        print(f"[{n}/{len(benzersiz)}] {metin[:90]}", flush=True)
        try:
            lab = tek_metin_etiketle(tokenizer, model, metin)
            cache[metin] = (lab["duygu"], ";".join(lab["konular"]))
        except Exception as exc:
            hatalar[metin] = str(exc)
            print(f"  HATA: {exc}", flush=True)
 
    rows = []
    for _, r in sample.iterrows():
        metin = str(r["metin"])
        if metin in cache:
            d, k = cache[metin]; h = ""
        else:
            d, k, h = "HATA", "", hatalar.get(metin, "Etiketleme başarısız.")
        rows.append({"geri_bildirim_id": r["geri_bildirim_id"], "metin": metin, "duygu": d, "konular": k, "hata": h})
    result = pd.DataFrame(rows)
    result.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nTamamlandı. Toplam: {len(result)} | Hatalı: {(result['duygu'] == 'HATA').sum()}\nDosya: {out_file}", flush=True)
 
 
def normalize_topics(value):
    if pd.isna(value):
        return set()
    return {t.strip() for t in str(value).split(";") if t.strip()}
 
 
def olc():
    gold_file, llm_file = C / "altin_set.csv", C / "llm_etiketleri.csv"
    for f in (gold_file, llm_file):
        if not f.exists():
            raise FileNotFoundError(f"{f.name} bulunamadı.")
    gold = pd.read_csv(gold_file, sep=None, engine="python", encoding="utf-8-sig")
    llm = pd.read_csv(llm_file, encoding="utf-8-sig")
    gold["geri_bildirim_id"] = gold["geri_bildirim_id"].astype(str)
    llm["geri_bildirim_id"] = llm["geri_bildirim_id"].astype(str)
    data = gold.merge(llm, on="geri_bildirim_id", how="inner", suffixes=("_el", "_model"), validate="one_to_one")
    for col in ("duygu_el", "konular_el"):
        data[col] = data[col].fillna("").astype(str).str.strip()
    ev = data[(data["duygu_el"] != "") & (data["konular_el"] != "") & (data["duygu_model"] != "HATA")].copy()
    if ev.empty:
        print("Değerlendirilecek kayıt yok. Altın seti ve model çıktısını kontrol edin.")
        return
    ev["duygu_dogru"] = ev["duygu_el"] == ev["duygu_model"]
    ev["konu_tam"] = ev.apply(lambda r: normalize_topics(r["konular_el"]) == normalize_topics(r["konular_model"]), axis=1)
    print(f"Değerlendirilen: {len(ev)} | Duygu doğruluğu: {ev['duygu_dogru'].mean():.3f} | Konu tam eşleşme: {ev['konu_tam'].mean():.3f}")
    print(pd.crosstab(ev["duygu_el"], ev["duygu_model"], rownames=["Gerçek"], colnames=["Model"]))
    ev[~ev["duygu_dogru"] | ~ev["konu_tam"]].to_csv(C / "llm_hatalari.csv", index=False, encoding="utf-8-sig")
    print("Hatalı satırlar: ciktilar/llm_hatalari.csv")
 
 
def main():
    parser = argparse.ArgumentParser(description="Hugging Face ile yerel geri bildirim etiketleme")
    parser.add_argument("islem", choices=["ornek", "etiketle", "olc"])
    parser.add_argument("--model", default=MODEL, help="Hugging Face model adı")
    parser.add_argument("--limit", type=int, default=None, help="Test için ilk N kayıt")
    args = parser.parse_args()
    if args.islem == "ornek":
        ornek()
    elif args.islem == "etiketle":
        etiketle(model_name=args.model, limit=args.limit)
    else:
        olc()
 
 
if __name__ == "__main__":
    main()