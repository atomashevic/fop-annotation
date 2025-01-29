from PIL import Image, ImageDraw

def draw_scores(
    frame: Image,
    emotions: dict,
    bounding_box: dict,
    size_multiplier: int = 1,
) -> Image:
    """Draw scores for each emotion under faces."""
    GRAY = (211, 211, 211)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    x, y, w, h = bounding_box
    max_score = max(emotions.values())

    for idx, (emotion, score) in enumerate(emotions.items()):
        color = GRAY if score < 0.01 else YELLOW
        if score == max_score:
            color = GREEN
        emotion_score = f"{emotion}: {score*100:.2f}%"
        frame.text((x + w//2, y + h + (10 * size_multiplier) + idx * (10 * size_multiplier)), emotion_score, fill =color, stroke_width=1)
    return frame        


def draw_annotations(
    frame: Image,
    faces: list,
    boxes=True,
    scores=True,
    size_multiplier: int = 1,
) -> Image:
    """Draws boxes around detected faces. Faces is a list of dicts with `box` and `emotions`."""
    if not faces:
        return frame
    frame = ImageDraw.Draw(frame)
    for face in faces:
        x, y, w, h = face["box"]
        emotions = face["emotions"]

        if boxes:
            shape = [(x, y), (x+ w, y + h)]
            frame.rectangle(shape, outline ="red", width=2)

        if scores:
            draw_scores(frame, emotions, (x, y, w, h), size_multiplier)