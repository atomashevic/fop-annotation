def map_mafw_to_sentiment(single_label):
    """Map MAFW single_label to sentiment categories
    
    Label mapping:
    1: anger, 2: disgust, 3: fear, 4: happiness, 5: neutral
    6: sadness, 7: surprise, 8: contempt, 9: anxiety
    10: helplessness, 11: disappointment
    """
    positive = {4, 7}  # happiness and surprise
    negative = {1, 2, 3, 6, 9, 10, 11}  # anger, disgust, fear, sadness, anxiety, helplessness, disappointment
    neutral = {5}  # neutral
    # contempt (8) remains unmapped as it's ambiguous
    
    if single_label in positive:
        return 'positive'
    elif single_label in negative:
        return 'negative'
    elif single_label in neutral:
        return 'neutral'
    else:
        return 'other'

def map_fer_to_sentiment(fer_results):
    """Map FER emotions to sentiment categories"""
    positive = ['happy0', 'surprise0']
    negative = ['angry0', 'disgust0', 'fear0', 'sad0']
    neutral = ['neutral0']
    
    sentiment_scores = {
        'positive': sum(fer_results[emotion] for emotion in positive),
        'negative': sum(fer_results[emotion] for emotion in negative),
        'neutral': sum(fer_results[emotion] for emotion in neutral)
    }
    return max(sentiment_scores.items(), key=lambda x: x[1])[0] 