import os, sys, cv2, glob
import pandas as pd

def main():
    # Load templates directly using cv2
    templates = []
    for p in glob.glob("logo_samples/positive/*.*"):
        img = cv2.imread(p, 0) # Grayscale
        if img is not None:
            templates.append(img)
            
    df = pd.read_csv("output/scraped_results.csv")
    
    results = []
    for _, row in df.iterrows():
        url = row['url']
        shot = row['screenshot']
        score = 0.0
        
        if isinstance(shot, str) and os.path.exists(shot):
            target = cv2.imread(shot, 0)
            if target is not None:
                for t in templates:
                    if t.shape[0] > target.shape[0] or t.shape[1] > target.shape[1]:
                        t = cv2.resize(t, (target.shape[1], target.shape[0]))
                    res = cv2.matchTemplate(target, t, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > score:
                        score = max_val
        
        risk = "HIGH" if score >= 0.70 else ("MEDIUM" if score >= 0.50 else "LOW")
        results.append({"url": url, "score": score, "risk": risk})
    
    res_df = pd.DataFrame(results).sort_values("score", ascending=False)
    
    print("\n\n=== IITM LOGO MISUSE WEBSITES ===")
    
    high = res_df[res_df["risk"] == "HIGH"]
    med = res_df[res_df["risk"] == "MEDIUM"]
    
    print(f"\n[ HIGH RISK - Likely Using Logo ] ({len(high)} found)")
    for _, r in high.iterrows():
        print(f"  {r['score']*100:.1f}% : {r['url']}")
        
    print(f"\n[ MEDIUM RISK - Possible Logo ] ({len(med)} found)")
    for _, r in med.iterrows():
        print(f"  {r['score']*100:.1f}% : {r['url']}")
        
    print(f"\nTotal scanned: {len(res_df)}")

if __name__ == "__main__":
    main()
