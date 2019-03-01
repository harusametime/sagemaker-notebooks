import tensorflow as tf
from PIL import Image
import numpy as np

    
def neo_preprocess(payload, content_type):
    import logging
    import numpy as np
    import PIL.Image   # Training container doesn't have this package
    import io

    logging.info('Invoking user-defined pre-processing function')

    if content_type != 'application/x-image':
        raise RuntimeError('Content type must be application/x-image')
    
    f = io.BytesIO(payload)
    # Load image and convert to greyscale space
    image = PIL.Image.open(f)
    # Resize
    image = np.asarray(image.resize((128, 128)))
    
    # Standardize 
    image = (image*(1.0/255) - 0.5)*2
    
    # Reshape
    image = image.reshape((1, 128,128,3)).astype('float32')

    return image

### NOTE: this function cannot use MXNet
def neo_postprocess(result):
    import logging
    import numpy as np
    import json

    logging.info('Invoking user-defined post-processing function')
    
    # Softmax (assumes batch size 1)
    result = np.squeeze(result)
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)

    response_body = json.dumps(result.tolist())
    content_type = 'application/json'

    return response_body, content_type