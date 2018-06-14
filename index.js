let weights_cache = {}
var SIZE = 256
async function fetch_weight(path) {
  return new Promise(function(resolve, reject) {
    if (path in weights_cache) {
      resolve(weights_cache[path])
      return
    }

    let xhr = new XMLHttpRequest()
    xhr.open("GET", path, true)
    xhr.responseType = "arraybuffer"

    xhr.onload = function(e) {
      if (xhr.status != 200) {
        reject("missing model")
        return
      }
      let buf = xhr.response
      if (!buf) {
        reject("invalid arraybuffer")
        return
      }

      let parts = []
      let offset = 0
      while (offset < buf.byteLength) {
        let b = new Uint8Array(buf.slice(offset, offset+4))
        offset += 4
        let len = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
        parts.push(buf.slice(offset, offset + len))
        offset += len
      }

      let shapes = JSON.parse((new TextDecoder("utf8")).decode(parts[0]))
      let index = new Float32Array(parts[1])
      let encoded = new Uint8Array(parts[2])

      // decode using index
      let arr = new Float32Array(encoded.length)
      for (let i = 0; i < arr.length; i++) {
        arr[i] = index[encoded[i]]
      }

      let weights = {}
      offset = 0
      for (let i = 0; i < shapes.length; i++) {
        let shape = shapes[i].shape
        let size = shape.reduce((total, num) => total * num)
        let values = arr.slice(offset, offset+size)
        let dlarr = dl.Array1D.new(values, "float32")
        weights[shapes[i].name] = dlarr.reshape(shape)
        offset += size
      }
      weights_cache[path] = weights
      resolve(weights)
    }
    xhr.send(null)
  })
}

function mlmodel(input, weights) {
  const math = dl.ENV.math

  function preprocess(input) {
    return math.subtract(math.multiply(input, dl.Scalar.new(2)), dl.Scalar.new(1))
  }

  function deprocess(input) {
    return math.divide(math.add(input, dl.Scalar.new(1)), dl.Scalar.new(2))
  }

  function batchnorm(input, scale, offset) {
    var moments = math.moments(input, [0, 1])
    const varianceEpsilon = 1e-5
    return math.batchNormalization3D(input, moments.mean, moments.variance, varianceEpsilon, scale, offset)
  }

  function conv2d(input, filter, bias) {
    return math.conv2d(input, filter, bias, [2, 2], "same")
  }

  function deconv2d(input, filter, bias) {
    var convolved = math.conv2dTranspose(input, filter, [input.shape[0]*2, input.shape[1]*2, filter.shape[2]], [2, 2], "same")
    var biased = math.add(convolved, bias)
    return biased
  }

  var preprocessed_input = preprocess(input)

  var layers = []

  var filter = weights["generator/encoder_1/conv2d/kernel"]
  var bias = weights["generator/encoder_1/conv2d/bias"]
  var convolved = conv2d(preprocessed_input, filter, bias)
  layers.push(convolved)

  for (var i = 2; i <= 8; i++) {
    var scope = "generator/encoder_" + i.toString()
    var filter = weights[scope + "/conv2d/kernel"]
    var bias = weights[scope + "/conv2d/bias"]
    var layer_input = layers[layers.length - 1]
    var rectified = math.leakyRelu(layer_input, 0.2)
    var convolved = conv2d(rectified, filter, bias)
    var scale = weights[scope + "/batch_normalization/gamma"]
    var offset = weights[scope + "/batch_normalization/beta"]
    var normalized = batchnorm(convolved, scale, offset)
    layers.push(normalized)
  }

  for (var i = 8; i >= 2; i--) {
    if (i == 8) {
      var layer_input = layers[layers.length - 1]
    } else {
      var skip_layer = i - 1
      var layer_input = math.concat3D(layers[layers.length - 1], layers[skip_layer], 2)
    }
    var rectified = math.relu(layer_input)
    var scope = "generator/decoder_" + i.toString()
    var filter = weights[scope + "/conv2d_transpose/kernel"]
    var bias = weights[scope + "/conv2d_transpose/bias"]
    var convolved = deconv2d(rectified, filter, bias)
    var scale = weights[scope + "/batch_normalization/gamma"]
    var offset = weights[scope + "/batch_normalization/beta"]
    var normalized = batchnorm(convolved, scale, offset)
    // missing dropout
    layers.push(normalized)
  }

  var layer_input = math.concat3D(layers[layers.length - 1], layers[0], 2)
  var rectified = math.relu(layer_input)
  var filter = weights["generator/decoder_1/conv2d_transpose/kernel"]
  var bias = weights["generator/decoder_1/conv2d_transpose/bias"]
  var convolved = deconv2d(rectified, filter, bias)
  var rectified = math.tanh(convolved)
  layers.push(rectified)

  var output = layers[layers.length - 1]
  var deprocessed_output = deprocess(output)

  return deprocessed_output
}

// Converts a tf to DOM img
const array3DToImage = (tensor) => {
  const [imgWidth, imgHeight] = tensor.shape;
  const data = tensor.dataSync();
  const canvas = document.createElement('canvas');
  canvas.width = imgWidth;
  canvas.height = imgHeight;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < imgWidth * imgHeight; i += 1) {
    const j = i * 4;
    const k = i * 3;
    imageData.data[j + 0] = Math.floor(256 * data[k + 0]);
    imageData.data[j + 1] = Math.floor(256 * data[k + 1]);
    imageData.data[j + 2] = Math.floor(256 * data[k + 2]);
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);

  // Create img HTML element from canvas
  const dataUrl = canvas.toDataURL();
  const outputImg = document.createElement('img');
  outputImg.src = dataUrl;
  outputImg.style.width = imgWidth;
  outputImg.style.height = imgHeight;
  return outputImg;
};

function setup() {
  fetch_weight('/models/edges2pikachu_AtoB.pict')
  .then(weights => {
    const math = dl.ENV.math
    let imgElement = document.getElementById('input')
    let input = dl.Array3D.fromPixels(imgElement)
    const normalized_input = math.divide(input, dl.Scalar.new(255.));
    let output_rgb = mlmodel(normalized_input, weights)
    let outputImg = array3DToImage(output_rgb);
    createImg(outputImg.src).parent('output');
  })
}
