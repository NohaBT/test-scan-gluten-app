import math
import re
from collections import Counter

import cv2
import pytesseract
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode


st.set_page_config(page_title="Gluten Scanner", page_icon="🔎")
st.warning("MVP note: OCR works best with clear English ingredient labels. Always verify the product label.")

TEXT = {
    "English": {
        "language": "Choose language",
        "subtitle": "Scan a barcode or ingredients list",
        "intro": "This tool uses rules + a small machine learning model to estimate gluten risk from a product photo.",
        "info": "Try scanning the barcode first. If the product is not found, scan the ingredients list.",
        "scan_type": "Choose scan type",
        "barcode": "Scan Barcode",
        "ingredients": "Scan Ingredients",
        "upload": "Upload photo",
        "tip": "Tip: use good lighting and keep the barcode or ingredients list clear.",
        "found_product": "Product found",
        "product_info": "Product Information",
        "unknown_product": "Unknown product",
        "unknown_brand": "Unknown brand",
        "not_found": "Product not found in the database. Choose 'Scan Ingredients' and upload the ingredients list.",
        "no_barcode": "No barcode detected. Please upload a clear barcode photo.",
        "show_text": "Show extracted text",
        "no_text": "No text detected.",
        "unclear": "Text is not clear enough. Please upload a clearer ingredients photo.",
        "result": "Analysis Result",
        "contains": "Contains Gluten",
        "possible": "Possible Gluten Risk",
        "free": "Likely Gluten Free",
        "detected": "Detected ingredients:",
        "ml": "Machine Learning Estimate",
        "probability": "Gluten probability",
        "risk": "Risk level",
        "high": "High Risk",
        "medium": "Medium Risk",
        "low": "Low Risk",
        "why": "Why this result?",
        "rule_reason": "Rule-based detection found gluten terms.",
        "ml_reason": "The ML model found a gluten-like pattern even without an exact keyword match.",
        "safe_reason": "A gluten-free phrase was detected, so the warning was reduced.",
        "caption": "This app helps with screening, but always confirm labels if you have celiac disease or severe allergy.",
        "recent": "Recent Scans",
    },
    "Français": {
        "language": "Choisir la langue",
        "subtitle": "Scannez un code-barres ou une liste d'ingrédients",
        "intro": "Cet outil utilise des règles + un petit modèle de machine learning pour estimer le risque de gluten.",
        "info": "Essayez d'abord le code-barres. Si le produit n'est pas trouvé, scannez les ingrédients.",
        "scan_type": "Choisir le type de scan",
        "barcode": "Scanner le code-barres",
        "ingredients": "Scanner les ingrédients",
        "upload": "Importer une photo",
        "tip": "Astuce : utilisez une bonne lumière et gardez le texte ou le code-barres bien visible.",
        "found_product": "Produit trouvé",
        "product_info": "Informations produit",
        "unknown_product": "Produit inconnu",
        "unknown_brand": "Marque inconnue",
        "not_found": "Produit introuvable. Choisissez 'Scanner les ingrédients' et importez la liste.",
        "no_barcode": "Aucun code-barres détecté. Importez une photo plus claire.",
        "show_text": "Afficher le texte extrait",
        "no_text": "Aucun texte détecté.",
        "unclear": "Le texte n'est pas assez clair. Importez une photo plus lisible.",
        "result": "Résultat de l'analyse",
        "contains": "Contient du gluten",
        "possible": "Risque de gluten possible",
        "free": "Probablement sans gluten",
        "detected": "Ingrédients détectés :",
        "ml": "Estimation Machine Learning",
        "probability": "Probabilité de gluten",
        "risk": "Niveau de risque",
        "high": "Risque élevé",
        "medium": "Risque moyen",
        "low": "Risque faible",
        "why": "Pourquoi ce résultat ?",
        "rule_reason": "La détection par règles a trouvé des termes liés au gluten.",
        "ml_reason": "Le modèle ML a détecté un motif proche du gluten sans mot-clé exact.",
        "safe_reason": "Une phrase sans gluten a été détectée, donc l'alerte a été réduite.",
        "caption": "Cette app aide au dépistage, mais confirmez toujours l'étiquette en cas de maladie cœliaque ou d'allergie sévère.",
        "recent": "Scans récents",
    },
    "العربية": {
        "language": "اختر اللغة",
        "subtitle": "صوّر الباركود أو لائحة المكونات",
        "intro": "هذا التطبيق يستعمل القواعد + نموذج تعلم آلي صغير لتقدير خطر الغلوتين.",
        "info": "  جرّب الباركود أولا. إذا لم نجد المنتج، صوّر لائحة المكونات.",
        "scan_type": "اختر نوع الفحص",
        "barcode": "فحص الباركود",
        "ingredients": "فحص المكونات",
        "upload": "ارفع الصورة",
        "tip": "نصيحة: استعمل إضاءة جيدة وحافظ على الباركود أو المكونات واضحة.",
        "found_product": "تم العثور على المنتج",
        "product_info": "معلومات المنتج",
        "unknown_product": "منتج غير معروف",
        "unknown_brand": "علامة غير معروفة",
        "not_found": "لم نجد المنتج في قاعدة البيانات. اختر فحص المكونات وارفع صورة اللائحة.",
        "no_barcode": "لم يتم اكتشاف الباركود. ارفع صورة أوضح.",
        "show_text": "إظهار النص المستخرج",
        "no_text": "لم يتم اكتشاف نص.",
        "unclear": "النص غير واضح بما يكفي. ارفع صورة أوضح.",
        "result": "نتيجة التحليل",
        "contains": "يحتوي على الغلوتين",
        "possible": "احتمال وجود الغلوتين",
        "free": "غالبا خال من الغلوتين",
        "detected": "المكونات المكتشفة:",
        "ml": "تقدير التعلم الآلي",
        "probability": "احتمال الغلوتين",
        "risk": "مستوى الخطر",
        "high": "خطر مرتفع",
        "medium": "خطر متوسط",
        "low": "خطر منخفض",
        "why": "لماذا هذه النتيجة؟",
        "rule_reason": "القواعد وجدت كلمات مرتبطة بالغلوتين.",
        "ml_reason": "نموذج التعلم الآلي وجد نمطا يشبه مكونات الغلوتين بدون كلمة مطابقة تماما.",
        "safe_reason": "تم اكتشاف عبارة خال من الغلوتين، لذلك تم تخفيض التحذير.",
        "caption": "هذا التطبيق يساعد في الفحص الأولي، لكن تأكد دائما من الملصق إذا كان عندك مرض السيلياك أو حساسية قوية.",
        "recent": "آخر الفحوصات",
    },
}

SAFE_PHRASES = [
    "contains no gluten",
    "no gluten",
    "gluten free",
    "without gluten",
    "sans gluten",
    "ne contient pas de gluten",
    "بدون غلوتين",
    "بدون جلوتين",
    "خال من الغلوتين",
    "خالي من الغلوتين",
]

DANGEROUS_GLUTEN = [
    "wheat",
    "barley",
    "rye",
    "malt",
    "triticale",
    "spelt",
    "bulgur",
    "semolina",
    "farina",
    "breadcrumbs",
    "gluten",
    "wheat starch",
    "wheat protein",
    "brewer yeast",
    "brewer's yeast",
    "blé",
    "blé tendre",
    "orge",
    "seigle",
    "épeautre",
    "epautre",
    "boulgour",
    "semoule",
    "farine de blé",
    "chapelure",
    "son de blé",
    "amidon de blé",
    "protéine de blé",
    "levure de bière",
    "levure de biere",
    "قمح",
    "شعير",
    "جاودار",
    "دقيق القمح",
    "سميد",
    "برغل",
    "غلوتين",
    "جلوتين",
    "نشا القمح",
    "بروتين القمح",
    "خميرة البيرة",
]

TRAINING_EXAMPLES = [
    ("wheat flour sugar palm oil malt extract barley gluten", 1),
    ("semolina durum wheat eggs salt", 1),
    ("barley malt cereal wheat starch", 1),
    ("farine de blé sucre huile levure gluten", 1),
    ("semoule de blé dur orge malt", 1),
    ("chapelure farine de ble amidon de ble", 1),
    ("قمح شعير دقيق القمح غلوتين", 1),
    ("سميد برغل نشا القمح", 1),
    ("rice flour corn starch potato starch sugar salt", 0),
    ("gluten free oats rice cocoa milk powder", 0),
    ("sans gluten farine de riz mais pomme de terre", 0),
    ("riz mais huile tournesol lait poudre cacao", 0),
    ("lentils chickpeas rice salt spices", 0),
    ("بدون غلوتين دقيق الأرز ذرة بطاطس", 0),
    ("أرز ذرة عدس حمص ملح توابل", 0),
    ("milk cocoa sugar hazelnut soy lecithin", 0),
]



def normalize_text(text):
    text = text.lower()
    text = text.replace("’", "'").replace("œ", "oe")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return re.findall(r"[\w\u0600-\u06FF']+", normalize_text(text), flags=re.UNICODE)


@st.cache_resource
def train_gluten_model():
    class_word_counts = {0: Counter(), 1: Counter()}
    class_doc_counts = Counter()
    vocabulary = set()

    for text, label in TRAINING_EXAMPLES:
        tokens = tokenize(text)
        class_word_counts[label].update(tokens)
        class_doc_counts[label] += 1
        vocabulary.update(tokens)

    return {
        "word_counts": class_word_counts,
        "doc_counts": class_doc_counts,
        "vocabulary": vocabulary,
        "total_docs": len(TRAINING_EXAMPLES),
    }


def predict_gluten_probability(text):
    model = train_gluten_model()
    tokens = tokenize(text)

    if not tokens:
        return 0.5

    scores = {}
    vocab_size = len(model["vocabulary"])

    for label in (0, 1):
        word_counts = model["word_counts"][label]
        total_words = sum(word_counts.values())
        score = math.log(model["doc_counts"][label] / model["total_docs"])

        for token in tokens:
            count = word_counts[token] + 1
            score += math.log(count / (total_words + vocab_size))

        scores[label] = score

    max_score = max(scores.values())
    gluten_score = math.exp(scores[1] - max_score)
    free_score = math.exp(scores[0] - max_score)
    return gluten_score / (gluten_score + free_score)


if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

language = st.selectbox(
    "Choose language / Choisir la langue / اختر اللغة",
    ["English", "Français", "العربية"],
)
t = TEXT[language]

st.title("🔎 Gluten Scanner")
st.markdown(f"### {t['subtitle']}")
st.write(t["intro"])
st.info(t["info"])

scan_mode = st.radio(t["scan_type"], [t["barcode"], t["ingredients"]])

uploaded_picture = st.file_uploader(
    t["upload"],
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

st.caption(t["tip"])


def is_safe_context(text, term):
    safe_nearby_words = [
        "no",
        "without",
        "free",
        "sans",
        "non",
        "بدون",
        "خال",
        "خالي",
    ]

    words = text.split()
    term_words = term.lower().split()

    for index in range(len(words)):
        window = words[index : index + len(term_words)]
        if " ".join(window) == " ".join(term_words):
            nearby = words[max(0, index - 4) : index]
            if any(neg in nearby for neg in safe_nearby_words):
                return True

    return False


def read_barcode(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    barcodes = decode(gray)

    if barcodes:
        return barcodes[0].data.decode("utf-8")

    return None


@st.cache_data
def get_product_from_openfoodfacts(barcode):

    url = f"https://world.openfoodfacts.net/api/v2/product/{barcode}.json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()

    if data.get("status") != 1:
        return None

    product = data.get("product", {})

    # DEBUG (temporary)
    # st.write(product)

    ingredients = (
        product.get("ingredients_text")
        or product.get("ingredients_text_en")
        or product.get("ingredients_text_fr")
        or product.get("ingredients_text_with_allergens")
        or ""
    )
    
    labels_text = " ".join([
        str(product.get("labels", "")),
        str(product.get("labels_tags", "")),
        str(product.get("_keywords", "")),
        str(product.get("traces", "")),
        str(product.get("traces_tags", "")),
    ])
    
    full_text = ingredients + " " + labels_text

    return {
        "name": product.get("product_name") or t["unknown_product"],
        "brand": product.get("brands") or t["unknown_brand"],
        "ingredients": full_text,
    }

def read_ingredients_from_image(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, None, fx=2, fy=2)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    text = pytesseract.image_to_string(gray, lang="eng+fra")
    return text


def analyze_gluten(text):
    normalized = normalize_text(text)

    safe_detected = any(
        phrase in normalized
        for phrase in SAFE_PHRASES
    )

    found = []

    for term in DANGEROUS_GLUTEN:
        normalized_term = normalize_text(term)

        if normalized_term in normalized:

            patterns = [
                f"no {normalized_term}",
                f"without {normalized_term}",
                f"free of {normalized_term}",
                f"gluten free",
                f"sans {normalized_term}",
            ]

            if any(p in normalized for p in patterns):
                continue

            found.append(term)

    ml_probability = predict_gluten_probability(text)

    if safe_detected:
        ml_probability *= 0.25

    return sorted(set(found)), safe_detected, ml_probability


def risk_level(found_gluten, ml_probability, safe_detected):
    if safe_detected:
        return t["low"]

    if found_gluten:
        if len(found_gluten) >= 2:
            return t["high"]
        return t["medium"]

    return t["low"]


barcode = None
product = None
file = uploaded_picture

if file:
    extracted_text = ""
    image = Image.open(file)
    image = ImageOps.exif_transpose(image)

    st.image(image, width=280)

    if scan_mode == t["barcode"]:
        barcode = read_barcode(image)
        st.write("DEBUG BARCODE:", barcode)

        if barcode:
            product = get_product_from_openfoodfacts(barcode)

            if product:
                st.markdown(
                    f"""
                    ### 📦 {t['product_info']}

                    - **Barcode:** {barcode}
                    - **Product:** {product["name"]}
                    - **Brand:** {product["brand"]}
                    """
                )
                extracted_text = product["ingredients"]
            else:
                st.warning(t["not_found"])
                st.stop()
        else:
            st.warning(t["no_barcode"])
            st.stop()

    elif scan_mode == t["ingredients"]:
        extracted_text = read_ingredients_from_image(image)

    with st.expander(t["show_text"]):
        st.write(extracted_text or t["no_text"])

    if len(extracted_text.split()) < 4:
        st.warning(
            "⚠️ Ingredients data is missing from the product database."
        )
        
        st.info(
            "Try scanning the ingredients label directly using Ingredients Scan."
        )
        
        st.stop()

    found_gluten, safe_detected, ml_probability = analyze_gluten(extracted_text)
    ml_percent = round(ml_probability * 100)
    risk = risk_level(found_gluten, ml_probability, safe_detected)
    final_risk = risk
    rule_detected = bool(found_gluten and not safe_detected)
    
    ml_detected = False
    
    if rule_detected:
        result_label = t["contains"]
    
    elif safe_detected:
        result_label = t["free"]
    
    elif ml_detected:
        result_label = t["possible"]
    
    else:
        result_label = t["free"]

    scan_result = {
        "product": product["name"] if barcode and product else t["unknown_product"],
        "result": result_label,
        "ml": f"{ml_percent}%",
    }

    if not st.session_state.scan_history or st.session_state.scan_history[-1] != scan_result:
        st.session_state.scan_history.append(scan_result)

    st.subheader(t["result"])

    if final_risk == t["high"]:
        st.error(result_label)
        
    elif final_risk == t["medium"]:
        st.warning(result_label)
        
    else:
        st.success(result_label)

    st.metric(t["risk"], final_risk)

    if safe_detected:
        final_risk = t["low"]
    
    with st.expander(t["why"], expanded=True):
    
        if safe_detected:
            st.success(
                "Safe phrases like 'gluten free' or 'no gluten' were detected."
            )
    
        elif found_gluten:
            st.error(
                f"Detected gluten-related ingredients: {', '.join(found_gluten)}"
            )
    
        elif ml_detected:
            st.warning(
                "The AI model detected patterns similar to gluten-containing products."
            )
    
        else:
            st.info(
                "No strong gluten indicators were detected."
            )

    if found_gluten:
        st.write(t["detected"])
        for item in found_gluten:
            st.write("-", item)

    st.caption(t["caption"])

    st.markdown("---")
    st.subheader(f"🕘 {t['recent']}")

    history = st.session_state.scan_history[-5:]
    for item in reversed(history):
        st.write(f"- {item['product']} -> {item['result']} ({t['ml']}: {item['ml']})")
