def build_model(num_classes=10, learning_rate=0.001):
    base = MobileNetV2(weights='imagenet', include_top=False)
    ...
    return model