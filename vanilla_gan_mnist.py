"""📚 Import Required Libraries"""

# Utilities
import numpy as np
import matplotlib.pyplot as plt

# Keras components
from tensorflow.keras import initializers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Fix random seed for reproducibility
np.random.seed(1000)

# Dimension of the noise vector (latent space)
random_dim = 100

"""📥 Load and Preprocess MNIST Dataset"""


def load_mnist_data():
    # Load MNIST dataset
    (x_train, _), (_, _) = mnist.load_data()
    # Normalize pixel values from [0, 255] → [-1, 1]
    x_train = (x_train.astype(np.float32) / 127.5) - 1
    # Flatten 28x28 → 784 (vector form)
    x_train = x_train.reshape(60000, 784)
    return (x_train, _), (_, _)


"""🧑‍🎨 Define Generator Model"""


def get_generator():
    generator = Sequential()

    # Input: random noise vector
    generator.add(Input(shape=(random_dim,)))
    generator.add(Dense(256, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    # Hidden layers
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    # Output layer: 784 (flattened 28x28 image), values in [-1, 1]
    generator.add(Dense(784, activation="tanh"))

    return generator


"""🕵️ Define Discriminator Model"""


def get_discriminator():
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    discriminator = Sequential()

    # Input: flattened image vector (784)
    discriminator.add(Input(shape=(784,)))
    discriminator.add(
        Dense(1024, kernel_initializer=initializers.RandomNormal(stddev=0.02))
    )
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    # Hidden layers
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    # Output: probability of being real/fake
    discriminator.add(Dense(1, activation="sigmoid"))
    discriminator.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return discriminator


"""🔗 Build Combined GAN Model"""


def get_combined_model(discriminator, random_dim, generator):
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    # Freeze discriminator weights when training the generator
    discriminator.trainable = False

    # Input: noise vector → generator → fake image → discriminator
    z = Input(shape=(random_dim,))
    img = generator(z)
    validity = discriminator(img)

    # Combined model: trains generator to fool the discriminator
    combined_model = Model(inputs=z, outputs=validity)
    combined_model.compile(loss="binary_crossentropy", optimizer=optimizer)
    return combined_model


"""🖼️ Helper Function: Plot Generated Images"""


def plot_generated_images(
    epoch, generator, examples=100, dim=(10, 10), fig_size=(10, 10)
):
    # Sample noise
    noise = np.random.normal(loc=0, scale=1, size=(examples, random_dim))
    generated_images = generator.predict(noise, verbose=0)
    generated_images = generated_images.reshape(examples, 28, 28)

    # Plot and save generated images
    plt.figure(figsize=fig_size)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        # Rescale [-1, 1] → [0, 255] for visualization
        img = (0.5 * generated_images[i] + 0.5) * 255
        img = img.astype(np.uint8)
        plt.imshow(img, interpolation="nearest", cmap="gray_r")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"vanilla_output/gan_generated_image_epoch_{epoch}.png")


"""🏋️ Training Function"""


def train(epochs=1, batch_size=128, save_interval=20):
    # Load training data
    (x_train, _), (_, _) = load_mnist_data()

    # Initialize GAN components
    generator = get_generator()
    discriminator = get_discriminator()
    combined_model = get_combined_model(discriminator, random_dim, generator)

    # Labels: real=1 (smoothed to 0.9), fake=0
    valid = np.ones(shape=(batch_size, 1)) * 0.9
    fake = np.zeros(shape=(batch_size, 1))

    for epoch in range(1, epochs + 1):

        # -------------------------
        #  Train Discriminator
        # -------------------------

        # Sample a random batch of real images
        idx = np.random.randint(0, x_train.shape[0], size=batch_size)
        imgs = x_train[idx]

        # Generate fake images
        noise = np.random.normal(loc=0, scale=1, size=(batch_size, random_dim))
        gen_imgs = generator.predict(noise, verbose=0)

        # Train discriminator (real=1, fake=0)
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(x=imgs, y=valid)
        d_loss_fake = discriminator.train_on_batch(x=gen_imgs, y=fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -------------------------
        #  Train Generator
        # -------------------------

        # Train generator to fool the discriminator
        discriminator.trainable = False
        noise = np.random.normal(loc=0, scale=1, size=(batch_size, random_dim))
        g_loss = combined_model.train_on_batch(x=noise, y=valid)

        # Print progress
        print(
            f"{epoch} [D loss: {d_loss[0]:.6f}, acc.: {d_loss[1]*100:.2f}%] "
            f"[G loss: {g_loss:.6f}]"
        )

        # Save models and generated samples at intervals
        if epoch == 1 or epoch % save_interval == 0:
            generator.save("vanilla_output/generator.keras")
            discriminator.save("vanilla_output/discriminator.keras")
            plot_generated_images(epoch, generator)


"""🚀 Train the Vanilla GAN"""

train(epochs=200, save_interval=50)
