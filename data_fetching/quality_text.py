"""Text-location evidence, deliberately not a semantic overlay classifier."""


def classify_text(polygon, width, height, face=None, subject=None):
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    box = [min(xs), min(ys), max(xs), max(ys)]

    def intersects(other):
        return bool(
            other
            and min(box[2], other[2]) > max(box[0], other[0])
            and min(box[3], other[3]) > max(box[1], other[1])
        )

    if intersects(face):
        label = "text_over_face_review_required"
    elif box[1] > height * 0.72 and box[2] - box[0] > width * 0.3:
        label = "possible_caption_or_lower_third"
    elif intersects(subject):
        label = "subject_text_or_overlay_ambiguous"
    else:
        label = "background_text_or_overlay_ambiguous"
    return {"box": box, "category": label, "semantic_status": "unverified"}
