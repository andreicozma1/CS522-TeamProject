from pygame.locals import *
import cv2
import pygame
import gzip
import pickle
from PreProcessing import PreProcessing
from convolution import *
import sys
import PIL
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from data_functions import min_max_scale_img
from matplotlib import pyplot as plt


def load_nn():
    # Load the best model from file
    with gzip.open('models/bpnn_best_model_numpy.pkl.gzip', 'rb') as f:
        nn = pickle.load(f)
        return nn


def t_resize(X):
    # resize training data
    return PreProcessing.resize_image(X, size=(128, 128))


def t_grayscale(X):
    return PreProcessing.img2grayscale(X)


def t_scale(X):
    return PreProcessing.min_max_scale_img(X)


def get_conv_gray(flatten):
    # convolution kernel
    k_outline = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # outline

    k_test1 = np.array([[-1, -1, -1],
                       [1.15, 1.15, 1.15],
                       [-1, -1, -1]])

    # build model
    model = Sequential()

    model.add(Conv(kernel=k_test1, name="input_layer"))
    model.add(Pooling2D(mode='max'))

    model.add(Conv(kernel=k_outline, name="input_layer"))
    model.add(Activation('relu'))

    model.add(Pooling2D(mode='max'))

    if (flatten):
        model.add(Flatten())

    return model


pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
surface = pygame.display.set_mode(
    [1280, 720], HWSURFACE | DOUBLEBUF | RESIZABLE)
# 0 Is the built in camera
cap = cv2.VideoCapture(0)
# Gets fps of your camera
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps:", fps)
# If your camera can achieve 60 fps
# Else just have this be 1-30 fps
cap.set(cv2.CAP_PROP_FPS, 30)

conv_flat = get_conv_gray(flatten=True)
conv_def = get_conv_gray(flatten=False)

nn = load_nn()


def preprocess_image(frame, blip_mode):
    img = frame.copy()

    left = (1280 / 2 - 300)
    top = (720 / 2 - 300)
    right = (1280 / 2 + 300)
    bottom = (720 / 2 + 300)

    img_cropped = np.asarray(Image.fromarray(
        img).crop((left, top, right, bottom)))

    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_resized = t_resize(img_gray)
    img_scaled = t_scale(img_resized)
    # img_scaled = img_resized

    img_conv = conv_def.feedforward(img_scaled)
    img_conv_flat = conv_flat.feedforward(img_scaled)

    if blip_mode == 1:
        return img_cropped, img_conv_flat
    elif blip_mode == 2:
        return img_gray, img_conv_flat
    elif blip_mode == 3:
        return img_resized, img_conv_flat
    elif blip_mode == 4:
        return img_scaled, img_conv_flat
    elif blip_mode == 5:
        return img_conv, img_conv_flat

    return None, img_conv_flat


def plot_image(img, title='Image'):
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def make_surface(frame, scale=None):
    frame = np.fliplr(frame)
    frame = np.rot90(frame)
    surf = pygame.surfarray.make_surface(frame)
    if scale is not None:
        surf = pygame.transform.scale(surf, scale)
    return surf


def make_text_info(pred, x=1280 / 2, y=50):
    # Show text in the middle of the screen with the image mode
    font = pygame.font.SysFont('Comic Sans MS', 30)
    text = font.render(pred, True, (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = (x, y)
    return text, textRect


def predict(x_flat):
    y_vec = nn.feedforward(np.reshape(
        np.asarray(x_flat), (900, 1)))
    print(y_vec)

    prediction = np.argmax(y_vec)

    print(f"# Prediction: {prediction}")
    if prediction:
        return "Not Wearing Mask"
    else:
        return "Wearing Mask"


blip_mode = 0
blip_size = 200
blip_size_min = 50
blip_size_max = 450
blip_size_step = 50

while True:
    surface.fill([0, 0, 0])

    success, frame = cap.read()
    if not success:
        break

    # for some reasons the frames appeared inverted

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    blip_img, img_conv_flat = preprocess_image(frame, blip_mode)

    pred = predict(img_conv_flat)

    surf_full = make_surface(frame)

    surf_blip = None
    if blip_img is not None:
        surf_blip = make_surface(blip_img, scale=(blip_size, blip_size))

    text, textRect = make_text_info(pred)

    for event in pygame.event.get():
        # print(event)
        if event.type == pygame.KEYUP:
            # Plot each stage of the image with number keys 1-9
            if event.key == pygame.K_1:
                blip_mode = 1
            if event.key == pygame.K_2:
                blip_mode = 2
            if event.key == pygame.K_3:
                blip_mode = 3
            if event.key == pygame.K_4:
                blip_mode = 4
            if event.key == pygame.K_5:
                blip_mode = 5
            if event.key == pygame.K_0:
                blip_mode = 0

            if event.key == pygame.K_EQUALS:
                if blip_size + blip_size_step < blip_size_max:
                    blip_size += blip_size_step
                    print(
                        f"Preview Size: {blip_size}")
                else:
                    print(
                        f"Preview Size: {blip_size} => Cannot go above {blip_size_max}")

            if event.key == pygame.K_MINUS:
                if blip_size - blip_size_step > blip_size_min:
                    blip_size -= blip_size_step
                    print(
                        f"Preview Size: {blip_size}")
                else:
                    print(
                        f"Preview Size: {blip_size} => Cannot go below {blip_size_min}")

            if event.key == pygame.K_BACKSPACE:
                print("Goodbye!")
                # Exit the program
                pygame.quit()
                sys.exit(0)
            pygame.display.update
            break

    # Show the PyGame surface!
    surface.blit(surf_full, (0, 0))
    if surf_blip is not None:
        surface.blit(surf_blip, (0, 0))
    surface.blit(text, textRect)
    pygame.display.flip()
