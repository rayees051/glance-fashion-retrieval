colors = [
    "black","white","red","blue","green","yellow",
    "brown","gray","grey","pink","orange","purple",
    "beige","gold","silver","navy"
]

clothing = [
    "shirt","t-shirt","tee","blazer","jacket",
    "hoodie","coat","raincoat","dress",
    "skirt","jeans","pants","trousers",
    "shorts","tie","suit","sweater",
    "cardigan","boots","shoes","heels",
    "sneakers","hat","cap"
]

scene_keywords = {
    "office":"office",
    "park":"park",
    "street":"street",
    "city":"city",
    "home":"home",
    "room":"home",
    "runway":"fashion show"
}

style_keywords = {
    "formal":"formal",
    "business":"formal",
    "casual":"casual",
    "weekend":"casual",
    "fashion":"fashion"
}

def parse_query(query):

    text = query.lower()

    q = {
        "colors":[c for c in colors if c in text],
        "clothing":[c for c in clothing if c in text],
        "scene":"unknown",
        "style":"unknown"
    }

    for k,v in scene_keywords.items():
        if k in text:
            q["scene"] = v
            break

    for k,v in style_keywords.items():
        if k in text:
            q["style"] = v
            break

    return q