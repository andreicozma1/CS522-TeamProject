import numpy as np
from activation_functions import get_activation_func


# ----------------------------------------------------

class Layer:
    def __init__(self, name):
        self.name = name
        self.output_shape = (0, 0, 0)


# ----------------------------------------------------

class Conv(Layer):
    def __init__(self, kernel: np.array, padding=0, strides=1, name="conv"):
        super().__init__(f'{name} (Conv{len(kernel.shape)}d)')
        self.kernel = kernel
        self.padding = padding
        self.strides = strides

    def feedforward(self, img):
        assert len(img.shape) == 3 and img.shape[2] != len(self.kernel.shape), "Number of kernel filters must be " \
                                                                               "equal to number of dimensions "

        # Shape of Output Convolution
        xOutput = int(((img.shape[0] - self.kernel.shape[0] + 2 * self.padding) / self.strides) + 1)
        yOutput = int(((img.shape[1] - self.kernel.shape[1] + 2 * self.padding) / self.strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if self.padding != 0:
            # left and right padding
            z0 = np.zeros((self.padding, img.shape[1], img.shape[2]), dtype=img.dtype)
            # top and bottom padding
            z1 = np.zeros((img.shape[0] + self.padding * 2, self.padding, img.shape[2]), dtype=img.dtype)
            # surround the image with padding pixels
            imagePadded = np.concatenate((z1, np.concatenate((z0, img, z0), axis=0), z1), axis=1)
        else:
            # if no padding just save the original image
            imagePadded = img

        # Iterate through image
        for y in range(img.shape[1]):
            # Exit Convolution if the kernel expands past the end of the image
            if y > img.shape[1] - self.kernel.shape[1]:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % self.strides != 0:
                continue
            for x in range(img.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > img.shape[0] - self.kernel.shape[0]:
                    break
                # if a stride step, continue
                if x % self.strides != 0:
                    continue
                # catch error if index out of bounds
                try:
                    # Only Convolve if x has moved by the specified Strides
                    output[x, y] = (self.kernel * imagePadded[x: x + self.kernel.shape[0],
                                                  y: y + self.kernel.shape[1]]).sum()
                except Exception as e:
                    break
        return output


# ----------------------------------------------------

class Pooling2D(Layer):
    def __init__(self, pool_size=2, stride=2, padding=0, mode='max', name="pooling_2d"):
        super().__init__(f'{name} (MaxPooling2D)')
        # save size of pool filter
        self.pool_size = pool_size
        self.padding = padding
        self.stride = stride
        # save type of pooling layer into a lambda function to eliminate if checking later on
        if mode == 'max':
            self.mode = lambda img: np.max(img, axis=(2, 3))
        elif mode == 'min':
            self.mode = lambda img: np.min(img, axis=(2, 3))
        elif mode == 'avg':
            self.mode = lambda img: np.mean(img, axis=(2, 3))
        else:
            raise Exception('Pooling mode not supported')

    def feedforward(self, img):
        assert len(img.shape) != 2, 'Non 2D image given to 2D pool'

        # Pad img of a 2d image
        img = np.pad(img, self.padding, mode='constant')

        # calculate windowed image parameters
        # height of output img
        output_h = (img.shape[0] - self.pool_size) // self.stride + 1
        # width of output img
        output_w = (img.shape[1] - self.pool_size) // self.stride + 1
        # size of window to apply to image
        window_shape = (output_h, output_w, self.pool_size, self.pool_size)
        # number of strides in each dimension
        window_strides = (self.stride * img.strides[0], self.stride * img.strides[1], img.strides[0], img.strides[1])
        # create a windowed view of the image
        windowed_img = np.lib.stride_tricks.as_strided(img, window_shape, window_strides)

        return self.mode(windowed_img)


# ----------------------------------------------------
class Pooling3D(Layer):
    def __init__(self, pool_size=2, stride=2, padding=0, mode='max', name="pooling_3d"):
        super().__init__(f'{name} (MaxPooling3D)')
        # save size of pool filter
        self.pool_size = pool_size
        self.padding = padding
        self.stride = stride
        # save type of pooling layer into a lambda function to eliminate if checking later on
        if mode == 'max':
            self.mode = lambda img: np.max(img, axis=(2, 3))
        elif mode == 'min':
            self.mode = lambda img: np.min(img, axis=(2, 3))
        elif mode == 'avg':
            self.mode = lambda img: np.mean(img, axis=(2, 3))
        else:
            raise Exception('Pooling mode not supported')

    def feedforward(self, img):
        if len(img.shape) != 3:
            raise Exception('Non 3D image given to 3D pool')

        # Pad img in 3 dimensions
        if self.padding != 0:
            z0 = np.zeros((self.padding, img.shape[1], img.shape[2]), dtype=img.dtype)
            z1 = np.zeros((img.shape[0] + self.padding * 2, self.padding, img.shape[2]), dtype=img.dtype)
            img = np.concatenate((z1, np.concatenate((z0, img, z0), axis=0), z1), axis=1)

        # calculate windowed image parameters
        # height of output img
        output_h = (img.shape[0] - self.pool_size) // self.stride + 1
        # width of output img
        output_w = (img.shape[1] - self.pool_size) // self.stride + 1
        # size of window to apply to image
        window_shape = (output_h, output_w, self.pool_size, self.pool_size, len(img.shape))
        # create a windowed view of the image
        window_strides = (
            self.stride * img.strides[0], self.stride * img.strides[1], img.strides[0], img.strides[1], img.strides[2])
        # create a windowed view of the image
        windowed_img = np.lib.stride_tricks.as_strided(img, window_shape, window_strides)

        return self.mode(windowed_img)


# ----------------------------------------------------

class Activation(Layer):
    def __init__(self, activation, name="activation"):
        super().__init__(f'{name} (Activation)')
        # check activation function
        if type(activation) == str:
            self.activation = get_activation_func(activation)
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception("Bad Activation Function Given")

    def feedforward(self, x: np.array):
        return self.activation(x)


# ----------------------------------------------------

class Flatten(Layer):
    def __init__(self, name="flatten"):
        super().__init__(f'{name} (Flatten)')

    def feedforward(self, x: np.array):
        return x.flatten()


# ----------------------------------------------------


class Debug(Layer):
    def __init__(self, name="debug"):
        super().__init__(f'{name} (Debug)')
        self.name = name

    def feedforward(self, x: np.array):
        print("=" * 30)
        print("Current Shape: ", x.shape)
        print("Output From Last Layer: \n", x)
        print("=" * 30)
        return x


# ----------------------------------------------------

# class to hold each layer
class Sequential:

    # constructor
    def __init__(self):
        self.layers = []

    # prints out details of each layer
    def summary(self):
        print("Model: Sequential\n")
        print("#    Layer")
        for i, l in enumerate(self.layers):
            print(f"{i:<4} {l.name}")

    # add a layer to the list
    def add(self, layer: Layer):
        self.layers.append(layer)

    # feed an image through all layers
    def feedforward(self, x):
        for l in self.layers:
            x = l.feedforward(x)
        return x
