from HYPER_SD.SD3alpha import generate_image_SD3
from HYPER_SD.FLUXalpha import generate_image_FLUX

def generate_image(user_input: str, prompt: str):
    if user_input == "SD3":
        image = generate_image_SD3(prompt)

    elif user_input == "FLUX":
        image = generate_image_FLUX(prompt)

    return image


if __name__ == "__main__":
    model_name = str(input("Enter your model: ")) # SD3, FLUX
    prompt = "A man and a woman warmly holding hands, smiling at each other, in a sunny park setting."
    image = generate_image(model_name, prompt)

    image.save("all-output/output.png")