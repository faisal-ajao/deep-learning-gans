"""📚 Import Required Libraries"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Run functions eagerly (useful for debugging with TensorFlow)
tf.config.run_functions_eagerly(True)

# Keras layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    BatchNormalization,
    Activation,
    Conv2D,
    ZeroPadding2D,
    LeakyReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

"""⚙️ Define Parameters"""

# Image dimensions (28x28 grayscale)
img_shape = (28, 28, 1)

# Dimension of random noise vector (latent space)
latent_dim = 100

# Number of channels (grayscale = 1)
channels = 1

"""🧵 Download and Visualize the Fashion-MNIST Dataset"""

# Load only the training set (we don't need labels for GAN training)
(training_data, _), (_, _) = fashion_mnist.load_data()


# Function to visualize pixel values on top of the image
def visualize_input(img, ax):
    ax.imshow(img, cmap="gray")
    thresh = img.max() / 2.5
    width, height = img.shape
    for x in range(width):
        for y in range(height):
            ax.annotate(
                text=str(round(img[x][y], 2)),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if img[x][y] < thresh else "black",
            )


# Display a sample image with pixel values
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
visualize_input(training_data[3343], ax)

"""🔄 Preprocess Dataset (Rescale to [-1, 1])"""

# Normalize pixel values from [0, 255] → [-1, 1]
x_train = (training_data / 127.5) - 1

# Expand dimensions to (28,28,1)
x_train = np.expand_dims(x_train, axis=3)
print(x_train.shape)

"""🧑‍🎨 Build Generator Model"""


def build_generator():
    # Input: random noise vector
    noise = Input(shape=(latent_dim,))

    # Fully connected layer to reshape noise into feature maps
    x = Dense(128 * 7 * 7, activation="relu")(noise)
    x = Reshape((7, 7, 128))(x)
    x = UpSampling2D()(x)  # Upsample to 14x14

    # Convolutional layers with batch normalization
    x = Conv2D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)
    x = UpSampling2D()(x)  # Upsample to 28x28

    x = Conv2D(64, kernel_size=3, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)

    # Final output layer (tanh → values in [-1, 1])
    x = Conv2D(channels, kernel_size=3, padding="same")(x)
    img = Activation("tanh")(x)

    return Model(inputs=noise, outputs=img)


"""🕵️ Build Discriminator Model"""


def build_discriminator():
    # Input: image (28x28x1)
    img = Input(shape=img_shape)

    # Convolutional layers with LeakyReLU and Dropout
    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(img)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)  # Maintain shape
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.25)(x)

    # Flatten and output probability (real/fake)
    x = Flatten()(x)
    validity = Dense(1, activation="sigmoid")(x)

    return Model(inputs=img, outputs=validity)


"""🔗 Build and Compile Combined GAN"""

# Define optimizers for generator & discriminator
opt_disc = Adam(learning_rate=0.0002, beta_1=0.5)
opt_gen = Adam(learning_rate=0.0002, beta_1=0.5)

# Initialize discriminator
discriminator = build_discriminator()
discriminator.summary()

# Initialize generator
generator = build_generator()
generator.summary()

# Compile discriminator separately
discriminator.compile(
    loss="binary_crossentropy", optimizer=opt_disc, metrics=["accuracy"]
)

# Create combined GAN (generator + discriminator)
z = Input(shape=(latent_dim,))
img = generator(z)

# Freeze discriminator for combined training
discriminator.trainable = False

# Discriminator predicts validity of generated image
validity = discriminator(img)

# Combined model trains generator to fool the discriminator
combined_model = Model(inputs=z, outputs=validity)
combined_model.compile(loss="binary_crossentropy", optimizer=opt_gen)

# Optionally recompile discriminator with new optimizer
discriminator.trainable = True
new_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
discriminator.compile(
    loss="binary_crossentropy", optimizer=new_optimizer, metrics=["accuracy"]
)

"""🖼️ Helper Function to Plot Generated Images"""


def plot_generated_images(
    epoch, generator, examples=100, dim=(10, 10), fig_size=(10, 10)
):
    # Generate random noise
    noise = np.random.normal(loc=0, scale=1, size=(examples, latent_dim))
    generated_images = generator.predict(noise, verbose=0)
    generated_images = generated_images.reshape(examples, 28, 28)

    # Plot and save generated images
    plt.figure(figsize=fig_size)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation="nearest", cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"dcgan_output/gan_generated_image_epoch_{epoch}.png")


"""🏋️ Define Training Function"""


def train(epochs, batch_size=128, save_interval=50):

    # Labels for real (1) and fake (0) images
    valid = np.ones(shape=(batch_size, 1))
    fake = np.zeros(shape=(batch_size, 1))

    for epoch in range(epochs):

        # -------------------------
        #  Train Discriminator
        # -------------------------

        # Sample a batch of real images
        idx = np.random.randint(0, x_train.shape[0], size=batch_size)
        imgs = x_train[idx]

        # Generate fake images
        noise = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)

        # Train discriminator on real and fake images
        d_loss_real = discriminator.train_on_batch(x=imgs, y=valid)
        d_loss_fake = discriminator.train_on_batch(x=gen_imgs, y=fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -------------------------
        #  Train Generator
        # -------------------------

        # Generator wants discriminator to classify generated images as real
        g_loss = combined_model.train_on_batch(x=noise, y=valid)

        # Print training progress
        print(
            f"{epoch} [D loss: {d_loss[0]:.6f}, acc.: {d_loss[1]*100:.2f}%] "
            f"[G loss: {g_loss:.6f}]"
        )

        # Save generated images at intervals
        if epoch % save_interval == 0:
            generator.save("dcgan_output/generator.keras")
            discriminator.save("dcgan_output/discriminator.keras")
            plot_generated_images(epoch, generator)


"""🚀 Train the GAN Model"""

train(epochs=20_000, batch_size=32, save_interval=500)
