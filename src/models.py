import segmentation_models_pytorch as smp


def build_unet(encoder_name, encoder_weights=None):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
    )
