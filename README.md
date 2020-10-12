# mirrorpad-tfjs

MirrorPad implementation for tensorflow-js. Tested on 4D tensors of a [fast arbitrary image style transfer model (magenta)](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2) from tf_hub using Tensorflow 2.3.1 and tfjs-v2.6.0.


# Installation

Simply copy-paste mirrorpad.js into your .html or .js file. Make sure to choose between REFLECT and SYMMETRIC modes of the layer:

```javascript
tf.registerOp('MirrorPad', mirrorPadSymmetric);
```

**OR**

```javascript
tf.registerOp('MirrorPad', mirrorPadReflect);
```
