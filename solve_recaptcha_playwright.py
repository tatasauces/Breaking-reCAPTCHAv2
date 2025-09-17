"""
This script provides a reusable API to solve reCAPTCHA v2 challenges using Playwright and a machine learning model.
It can be used as a standalone script to solve the reCAPTCHA on the demo page, or it can be imported as a library
into other Playwright scripts.

To use it as a library, import the `solve_recaptcha_on_page` function:

    from solve_recaptcha_playwright import solve_recaptcha_on_page

Then, in your async Playwright script, call this function with your `page` object when a reCAPTCHA challenge is visible:

    is_solved = await solve_recaptcha_on_page(page)

The function will return `True` if the captcha is solved successfully, and `False` otherwise.
"""
import asyncio
from playwright.async_api import async_playwright
import os
import requests
from PIL import Image
from io import BytesIO
from models.YOLO_Classification import predict
from models.YOLO_Segment import predict as predict_segment
import time
import csv
from datetime import datetime
from IP import vpn
import traceback


# Constants
CAPTCHA_URL = "https://www.google.com/recaptcha/api2/demo"
THRESHOLD = 0.2
USE_TOP_N_STRATEGY = False
N = 3
YOLO_CLASSES = predict.get_class_names()
CHINESE_TO_ENGLISH_MAPPING = {
    "公車": "bus",
    "巴士": "bus",
    "行人穿越道": "crosswalk",
    "斑馬線": "crosswalk",
    "消防栓": "hydrant",
    "腳踏車": "bicycle",
    "自行車": "bicycle",
    "機車": "motorcycle",
    "機車/腳踏車": "motorcycle", # Mapping to motorcycle for now
    "汽車": "car",
    "橋": "bridge",
    "橋樑": "bridge",
    "煙囪": "chimney",
    "棕櫚樹": "palm",
    "樹木": "palm", # Assuming tree maps to palm
    "樓梯": "stairs",
    "梯子": "stairs", # Assuming ladder maps to stairs
    "紅綠燈": "traffic",
    "交通號誌": "traffic",
}
TYPE1 = True #one time image selection
TYPE2 = True #segmentation problem
TYPE3 = True #dynamic captcha
ENABLE_LOGS = True
ENABLE_VPN = False
ENABLE_MOUSE_MOVEMENT = True
ENABLE_NATURAL_MOUSE_MOVEMENT = True
ENABLE_COOKIES = True

def set_variables(variables):
    global CAPTCHA_URL, THRESHOLD, USE_TOP_N_STRATEGY, N, YOLO_CLASSES, TYPE1, TYPE2, TYPE3, ENABLE_LOGS, ENABLE_VPN, ENABLE_MOUSE_MOVEMENT, ENABLE_NATURAL_MOUSE_MOVEMENT, ENABLE_COOKIES
    if 'CAPTCHA_URL' in variables:
        CAPTCHA_URL = variables['CAPTCHA_URL']
    if 'THRESHOLD' in variables:
        THRESHOLD = variables['THRESHOLD']
    if 'USE_TOP_N_STRATEGY' in variables:
        USE_TOP_N_STRATEGY = variables['USE_TOP_N_STRATEGY']
    if 'N' in variables:
        N = variables['N']
    if 'YOLO_CLASSES' in variables:
        YOLO_CLASSES = variables['YOLO_CLASSES']
    if 'TYPE1' in variables:
        TYPE1 = variables['TYPE1']
    if 'TYPE2' in variables:
        TYPE2 = variables['TYPE2']
    if 'TYPE3' in variables:
        TYPE3 = variables['TYPE3']
    if 'ENABLE_LOGS' in variables:
        ENABLE_LOGS = variables['ENABLE_LOGS']
    if 'ENABLE_VPN' in variables:
        ENABLE_VPN = variables['ENABLE_VPN']
    if 'ENABLE_MOUSE_MOVEMENT' in variables:
        ENABLE_MOUSE_MOVEMENT = variables['ENABLE_MOUSE_MOVEMENT']
    if 'ENABLE_NATURAL_MOUSE_MOVEMENT' in variables:
        ENABLE_NATURAL_MOUSE_MOVEMENT = variables['ENABLE_NATURAL_MOUSE_MOVEMENT']
    if 'ENABLE_COOKIES' in variables:
        ENABLE_COOKIES = variables['ENABLE_COOKIES']

# Check if data dir is present
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)




COUNT = 0

# Global variable to store the log filename for the current session
log_filename = None
session_folder = None

def log(captcha_type, captcha_object):
    global log_filename, session_folder

    if not ENABLE_LOGS:
        return

    # If a session folder doesn't exist, create one
    if session_folder is None:
        # Find the highest existing session number
        highest_session_number = 0
        for dirname in os.listdir('.'):
            if dirname.startswith('Session'):
                try:
                    session_number = int(dirname[7:])
                    highest_session_number = max(highest_session_number, session_number)
                except ValueError:
                    # Ignore directories that don't have a number after "Session"
                    pass

        # Create a new session folder with a number one higher than the highest existing session number
        session_folder = f'Session{highest_session_number + 1:02}'
        os.makedirs(session_folder, exist_ok=True)

        # Save the current values of all global variables to a text file in the session folder
        save_global_variables()

    # Find the highest existing log file number
    highest_log_number = 0
    if log_filename is None:
        for filename in os.listdir(session_folder):
            if filename.startswith('logs_'):
                try:
                    log_number = int(filename[5:7])
                    highest_log_number = max(highest_log_number, log_number)
                except ValueError:
                    # Ignore files that don't have a number after "logs_"
                    pass

        # Create a new log file with a number one higher than the highest existing log file number
        log_filename = os.path.join(session_folder, f'logs_{highest_log_number + 1:02}.csv')

    with open(log_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Get the current time and format it as a string
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, captcha_type, captcha_object])



def save_global_variables():
    with open(os.path.join(session_folder, 'global_variables.txt'), 'w') as file:
        file.write(f'CAPTCHA_URL = {CAPTCHA_URL}\n')
        file.write(f'THRESHOLD = {THRESHOLD}\n')
        file.write(f'YOLO_CLASSES = {YOLO_CLASSES}\n')
        file.write(f'TYPE1 = {TYPE1}\n')
        file.write(f'TYPE2 = {TYPE2}\n')
        file.write(f'TYPE3 = {TYPE3}\n')
        file.write(f'ENABLE_LOGS = {ENABLE_LOGS}\n')
        file.write(f'ENABLE_VPN = {ENABLE_VPN}\n')
        file.write(f'ENABLE_MOUSE_MOVEMENT = {ENABLE_MOUSE_MOVEMENT}\n')
        file.write(f'ENABLE_NATURAL_MOUSE_MOVEMENT = {ENABLE_NATURAL_MOUSE_MOVEMENT}\n')
        file.write(f'ENABLE_COOKIES = {ENABLE_COOKIES}\n')
        file.write(f'USE_TOP_N_STRATEGY = {USE_TOP_N_STRATEGY}\n')
        file.write(f'N = {N}\n')



def reset_globals():
    global log_filename
    log_filename = None


async def open_browser_with_captcha(playwright):
    """
    Launches a browser, navigates to the reCAPTCHA demo page, and handles the initial checkbox click.

    Note: Playwright's support for Firefox profiles is limited. To enable cookie-based persistence
    (similar to the original script's use of a Firefox profile), this function uses Chromium
    when ENABLE_COOKIES is True, as it supports persistent contexts. When ENABLE_COOKIES is False,
    it uses Firefox to remain consistent with the original script's browser choice.
    """
    if ENABLE_VPN:
        vpn.connect()
        print("VPN connected")

    if ENABLE_COOKIES:
        print("Using Chromium for persistent cookies.")
        user_data_dir = os.path.join(os.getcwd(), "playwright_user_data")
        context = await playwright.chromium.launch_persistent_context(user_data_dir, headless=False)
        page = await context.new_page()
        browser = None # The context object handles closing the browser
    else:
        print("Using Firefox.")
        browser = await playwright.firefox.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

    try:
        await page.goto(CAPTCHA_URL, wait_until="networkidle")

        for _ in range(10):
            try:
                recaptcha_frame_locator = page.frame_locator("iframe[title='reCAPTCHA']")
                await recaptcha_frame_locator.locator(".recaptcha-checkbox-border").click(timeout=5000)

                # Wait for the challenge iframe to become visible
                challenge_frame_selector = "iframe[src*='bframe']"
                await page.wait_for_selector(challenge_frame_selector, state="visible", timeout=10000)

                print("Opened the browser with the captcha.")
                return page, context, browser
            except Exception:
                print("An error occurred. Reloading the page and trying again.")
                traceback.print_exc()
                await page.reload(wait_until="networkidle")

        print("Failed to open the browser with the captcha after 10 attempts.")
        await context.close()
        if browser:
            await browser.close()
        return None, None, None

    except Exception as e:
        print(f"A critical error occurred: {e}")
        traceback.print_exc()
        await context.close()
        if browser:
            await browser.close()
        return None, None, None


async def click_element(locator):
    """
    Clicks on a Playwright Locator.
    This is a simplified version of the original click_element function.
    """
    await locator.click()

async def process_tile(page, i, captcha_object_text, class_index):
    """
    Processes a single tile of the reCAPTCHA challenge.
    Takes a screenshot of the tile, classifies it using the ML model, and clicks on it if it matches the criteria.
    """
    global COUNT
    print(f"Processing tile with class index {class_index}")

    challenge_frame_locator = page.frame_locator('iframe[src*="bframe"]')

    xpath = f"//td[@id='{i}']"
    tile_locator = challenge_frame_locator.locator(xpath)

    filename = f"tile_{COUNT}.jpg"
    screenshot_path = os.path.join(data_dir, filename)
    await tile_locator.screenshot(path=screenshot_path, animations="disabled")

    result = predict.predict_tile(screenshot_path)
    predicted_index = result[2]
    if predicted_index >= len(YOLO_CLASSES):
        print(f"Warning: Model returned an out-of-bounds index: {predicted_index}")
        object_name = "other"
    else:
        object_name = YOLO_CLASSES[predicted_index]
    current_object_probability = result[0][class_index]

    # rename image
    os.rename(screenshot_path, os.path.join(data_dir, f"{object_name}_{filename}"))

    print(f"{COUNT}: The AI predicted tile to be {object_name} and probability is {current_object_probability}")

    COUNT += 1
    if USE_TOP_N_STRATEGY:
        top_n_indices = sorted(range(len(result[0])), key=lambda i: result[0][i], reverse=True)[:N]
        if class_index in top_n_indices:
            await click_element(tile_locator)
            return True
    else:
        if current_object_probability > THRESHOLD:
            print(f"{current_object_probability} > {THRESHOLD}")
            await click_element(tile_locator)
            return True

    return False


async def solve_type2(page):
    """
    Solves the 4x4 segmentation-based reCAPTCHA challenge.
    """
    save_path = "temp"
    os.makedirs(save_path, exist_ok=True)

    challenge_frame_locator = page.frame_locator('iframe[src*="bframe"]')

    xpath_image = "/html/body/div/div/div[2]/div[2]/div/table/tbody/tr[1]/td[1]/div/div[1]/img"
    xpath_text = "/html/body/div/div/div[2]/div[1]/div[1]/div/strong"

    captcha_text_locator = challenge_frame_locator.locator(f"xpath={xpath_text}")
    captcha_text = (await captcha_text_locator.inner_text()).strip()

    log("Type2", captcha_text)

    class_index = get_class_index(captcha_text)

    if class_index == -1:
        print(f"Could not find class index for captcha text: {captcha_text}")
        return

    img_locator = challenge_frame_locator.locator(f"xpath={xpath_image}")
    img_url = await img_locator.get_attribute("src")

    response = requests.get(img_url, stream=True)
    if response.status_code == 200:
        timestamp = str(time.time())
        filename = f"image_{captcha_text}_{timestamp}.png"
        filepath = os.path.join(save_path, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)

        success, grid = predict_segment.predict(class_index, filepath)

        xpath_tiles = "/html/body/div/div/div[2]/div[2]/div/table/tbody"
        tiles_to_click = [(i + 1, j + 1) for i in range(4) for j in range(4) if grid[i][j] == 1]

        for i, j in tiles_to_click:
            tile_locator = challenge_frame_locator.locator(f"xpath={xpath_tiles}/tr[{i}]/td[{j}]")
            await click_element(tile_locator)
            await page.wait_for_timeout(500) # a short delay

    verify_button_locator = challenge_frame_locator.locator("#recaptcha-verify-button")
    await click_element(verify_button_locator)
    await page.wait_for_timeout(500)


def get_class_index(captcha_object_text):
    """
    Gets the class index for a given captcha object text.
    Handles both English and Chinese labels.
    """
    # First, check for a direct match in the Chinese mapping to get the English name
    english_class_name = CHINESE_TO_ENGLISH_MAPPING.get(captcha_object_text)

    # If we have a mapped English name, find its index
    if english_class_name:
        for index, name in YOLO_CLASSES.items():
            if name.lower() == english_class_name.lower():
                return index
        return -1 # Mapped name not found in model's classes

    # If no direct Chinese match, search for an English class name as a substring
    # This handles cases like "Select all images with cars"
    for index, name in YOLO_CLASSES.items():
        if name.lower() in captcha_object_text.lower():
            return index

    return -1

async def captcha_is_solved(page):
    """
    Checks if the reCAPTCHA challenge is solved.
    """
    await page.wait_for_timeout(3000)
    try:
        recaptcha_frame_locator = page.frame_locator("iframe[title='reCAPTCHA']")
        checkbox_locator = recaptcha_frame_locator.locator("#recaptcha-anchor")
        aria_checked = await checkbox_locator.get_attribute('aria-checked')
        if aria_checked == 'true':
            print("captcha is solved")
            return True
        else:
            print("captcha is not solved yet")
            return False
    except Exception:
        return False

async def handle_dynamic_captcha(page, captcha_object_text, class_index, to_check):
    """
    Handles the dynamic reCAPTCHA challenges where new images appear after a correct selection.
    """
    challenge_frame_locator = page.frame_locator('iframe[src*="bframe"]')

    if not to_check:
        verify_button_locator = challenge_frame_locator.locator("#recaptcha-verify-button")
        await click_element(verify_button_locator)
        await page.wait_for_timeout(1000)
        return

    while True:
        await page.wait_for_timeout(2000)

        clicked_in_iteration = False

        indices_to_remove = []
        for i in to_check:
            if await process_tile(page, i, captcha_object_text, class_index):
                clicked_in_iteration = True
            else:
                indices_to_remove.append(i)

        for i in indices_to_remove:
            if i in to_check:
                to_check.remove(i)

        if not clicked_in_iteration:
            break

    verify_button_locator = challenge_frame_locator.locator("#recaptcha-verify-button")
    await click_element(verify_button_locator)
    await page.wait_for_timeout(2000)

    try:
        error_message_locator = challenge_frame_locator.locator(".rc-imageselect-error-select-more")
        if await error_message_locator.is_visible():
            print("The 'select more images' text appeared.")
            reload_button_locator = challenge_frame_locator.locator("#recaptcha-reload-button")
            await click_element(reload_button_locator)
    except Exception:
        print("The 'select more images' text did not appear.")


async def solve_classification_type(page, dynamic_captcha):
    """
    Solves the classification-based reCAPTCHA challenges (3x3 grid).
    """
    challenge_frame_locator = page.frame_locator('iframe[src*="bframe"]')

    captcha_object_locator = challenge_frame_locator.locator('#rc-imageselect strong')
    captcha_object_text = (await captcha_object_locator.inner_text()).strip()

    class_index = get_class_index(captcha_object_text)
    if class_index == -1:
        print(f"Could not find class index for: {captcha_object_text}")
        await page.wait_for_timeout(3000) # Wait before reloading
        reload_button_locator = challenge_frame_locator.locator("#recaptcha-reload-button")
        await click_element(reload_button_locator)
        return

    # Wait for the image grid to be visible before proceeding
    try:
        await challenge_frame_locator.locator(".rc-imageselect-table-33").wait_for(timeout=10000)
    except Exception as e:
        print(f"Error waiting for image grid: {e}")
        return

    if dynamic_captcha:
        log("dynamic", captcha_object_text)
    else:
        log("Type1", captcha_object_text)

    to_check = []
    for i in range(9):
        if await process_tile(page, i, captcha_object_text, class_index):
            to_check.append(i)

    if dynamic_captcha:
        await handle_dynamic_captcha(page, captcha_object_text, class_index, to_check)
    else:
        verify_button_locator = challenge_frame_locator.locator("#recaptcha-verify-button")
        await click_element(verify_button_locator)
        await page.wait_for_timeout(2000) # Add a wait after clicking verify

async def solve_recaptcha_on_page(page):
    """
    Solves the reCAPTCHA challenge on the given Playwright page.
    Assumes that the page is already at a state where a reCAPTCHA challenge is visible.

    :param page: The Playwright page object.
    :return: True if the captcha is solved successfully, False otherwise.
    """
    while True:
        try:
            challenge_frame_locator = page.frame_locator('iframe[src*="bframe"]')

            # Wait for either grid to be ready
            try:
                await challenge_frame_locator.locator('.rc-imageselect-table-33, .rc-imageselect-table-44').first.wait_for(timeout=10000)
            except Exception:
                print("No image grid found. Reloading.")
                await page.wait_for_timeout(3000) # Wait before reloading
                reload_button_locator = challenge_frame_locator.locator("#recaptcha-reload-button")
                await click_element(reload_button_locator)
                continue

            # Now check which one is visible
            is_4x4 = await challenge_frame_locator.locator(".rc-imageselect-table-44").is_visible()

            if is_4x4 and TYPE2:
                print("found a 4x4 segmentation problem")
                await solve_type2(page)
            else: # It should be a 3x3 grid
                imageselect_text = await challenge_frame_locator.locator('#rc-imageselect').inner_text()

                if "none" in imageselect_text and TYPE3:
                    print("found a 3x3 dynamic captcha")
                    await solve_classification_type(page, True)
                elif TYPE1:
                    print("found a 3x3 one time selection captcha")
                    await solve_classification_type(page, False)
                else:
                    await page.wait_for_timeout(3000) # Wait before reloading
                    reload_button_locator = challenge_frame_locator.locator("#recaptcha-reload-button")
                    await click_element(reload_button_locator)
                    continue

            if await captcha_is_solved(page):
                log("SOLVED", "captcha solved")
                return True
        except Exception as e:
            print("error occurred:", e)
            traceback.print_exc()
            if await captcha_is_solved(page):
                log("SOLVED", "captcha solved")
                return True
            try:
                await page.wait_for_timeout(3000) # Wait before reloading
                challenge_frame_locator = page.frame_locator('iframe[src*="bframe"]')
                reload_button_locator = challenge_frame_locator.locator("#recaptcha-reload-button")
                await click_element(reload_button_locator)
            except Exception as reload_e:
                print(f"Could not reload captcha: {reload_e}")
                return False
            continue
    return False


async def run_async():
    """
    The main asynchronous function to run the reCAPTCHA solver.
    This function serves as a demo of the `solve_recaptcha_on_page` API.
    """
    async with async_playwright() as p:
        page, context, browser = await open_browser_with_captcha(p)
        if not page:
            return

        is_solved = await solve_recaptcha_on_page(page)

        if is_solved:
            print("Captcha solved successfully by the API.")
        else:
            print("Failed to solve the captcha using the API.")

        if ENABLE_VPN:
            vpn.disconnect()
        await context.close()
        if browser:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(run_async())
