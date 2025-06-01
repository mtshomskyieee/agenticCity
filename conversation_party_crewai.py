import pygame
import random
import os
import csv
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
import openai
from openai import OpenAI
from crewai import Agent, Task, Crew, LLM, Process
import emoji
from crewai_tools import tool
import requests
from io import BytesIO
import json
from datetime import datetime

## Locally set the openai key
#os.environ["OPENAI_API_KEY"] = "<add your key here>"

## Turn off telemetry to 'telemetry.crewai.com`
os.environ["OTEL_SDK_DISABLED"] = "true"


# Initialize Pygame and Faker
pygame.init()
fake = Faker()

# Screen dimensions and other constants
GRID_SIZE = 10
CELL_SIZE = 40
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE * 2, GRID_SIZE * CELL_SIZE
FPS = 30
MAX_PLAYERS = 5  # Maximum number of players supported
SCROLL_SPEED = 5
RESPONSE_BOX_HEIGHT = 100


# Display setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Conversation Party")
clock = pygame.time.Clock()
WRAPPED_LINES = ""
# Global variable to track chatting state
is_chatting = False

# Add these imports at the top of the file if they're not already there
import os
from PIL import Image
import requests
from io import BytesIO
from openai import OpenAI


def ensure_asset_exists(asset_name: str, assets_dir: str) -> str:
    # Checks if an asset exists, if not generates it using OpenAI.
    # Returns the path to the asset.

    # Remove 'assets/' from the start of asset_name if it exists
    if asset_name.startswith('assets/'):
        asset_name = asset_name[7:]  # Remove 'assets/' prefix
        
    # Create full path
    asset_path = os.path.join(assets_dir, asset_name)
    
    # Create assets directory if it doesn't exist
    os.makedirs(assets_dir, exist_ok=True)
    
    # If asset already exists, return its path
    if os.path.exists(asset_path):
        return asset_path
        
    # If asset doesn't exist, generate it
    try:
        client = OpenAI()
        
        # Generate image using DALL-E
        response = client.images.generate(
            model="dall-e-3",  # Use DALL-E 3 for better quality
            prompt=f"Create a simple pixel art style icon for {asset_name.split('.')[0]}. The image should be minimal and clear, suitable for a 31x39 pixel game asset.",
            size="1024x1024",  # We'll resize it later
            n=1,
        )
        
        # Download the generated image
        image_url = response.data[0].url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Resize to 31x39 pixels
        img = img.resize((31, 39), Image.Resampling.LANCZOS)
        
        # Save the image
        img.save(asset_path)
        return asset_path
        
    except Exception as e:
        print(f"Failed to generate image for {asset_name}: {str(e)}")
        return None


def load_image(filename: str, size=(31,39)) -> pygame.Surface:
    # Load an image from the assets directory.
    try:
        # Remove 'assets/' from the start of filename if it exists
        if filename.startswith('assets/'):
            filename = filename[7:]  # Remove 'assets/' prefix
            
        # Create full path
        asset_path = os.path.join(assets_dir, filename)
        
        # Create assets directory if it doesn't exist
        os.makedirs(assets_dir, exist_ok=True)
        
        # If asset already exists, load it directly
        if os.path.exists(asset_path):
            image = pygame.image.load(asset_path)
            return pygame.transform.scale(image, size)
            
        # If asset doesn't exist, generate it
        try:
            client = OpenAI()  # Make sure OPENAI_API_KEY is set in environment variables
            
            # Generate image using DALL-E
            response = client.images.generate(
                model="dall-e-3",  # Use DALL-E 3 for better quality
                prompt=f"Create a simple pixel art style icon for {filename.split('.')[0]}. The image should be minimal and clear, suitable for a {size[0]}x{size[1]} pixel game asset.",
                size="1024x1024",  # We'll resize it later
                n=1,
            )
            
            # Download the generated image
            image_url = response.data[0].url
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            
            # Resize to target size
            img = img.resize(size, Image.Resampling.LANCZOS)
            
            # Save the image
            img.save(asset_path)
            
            # Convert to pygame surface and return
            return pil_to_pygame(img)
            
        except Exception as e:
            print(f"Failed to generate image for {filename}: {str(e)}")
            # Create a default surface with magenta color to indicate missing texture
            surface = pygame.Surface(size)
            surface.fill((255, 0, 255))  # Magenta
            return surface
            
    except Exception as e:
        print(f"Error loading image {filename}: {str(e)}")
        # Create a default surface with magenta color to indicate missing texture
        # If you're getting magenta surfaces it could be you cannot produce images with your OpenAI key
        surface = pygame.Surface(size)
        surface.fill((255, 0, 255))  # Magenta
        return surface


# Load and resize images
assets_dir = 'assets'
grass_img = load_image('grass.png', (CELL_SIZE, CELL_SIZE))
rock_img = load_image('rock.png', (CELL_SIZE, CELL_SIZE))
water_img = load_image('water.png', (CELL_SIZE, CELL_SIZE))
player_imgs = [load_image(f'player_{i}.png', (CELL_SIZE, CELL_SIZE)) for i in range(5)]


def pil_to_pygame(pil_image: Image.Image) -> pygame.Surface:
    # Convert a PIL Image to a Pygame Surface.

    # Ensure the image is in RGBA mode.
    pil_image = pil_image.convert("RGBA")
    mode = pil_image.mode
    size = pil_image.size
    data = pil_image.tobytes()
    return pygame.image.fromstring(data, size, mode)

def generate_emoji_image(emoji_name: str) -> Image.Image:
    """
    Generates a 40x40 image with an emoji based on the name provided.

    Args:
        emoji_name (str): The name of the emoji ('truck', 'car', etc.)

    Returns:
        PIL.Image.Image: An image of size 40x40 with the emoji drawn in the center.
    """
    # Define a simple mapping from emoji names to actual emoji characters.
    emoji_map = {
        # Transport
        "truck": "🚚",
        "car": "🚗",
        "taxi": "🚕",
        "bus": "🚌",
        "bicycle": "🚲",
        "motorbike": "🏍️",
        "airplane": "✈️",
        "ship": "🚢",
        "rocket": "🚀",

        # Faces and Emotions
        "smile": "😊",
        "grin": "😁",
        "wink": "😉",
        "laugh": "😂",
        "sad": "😢",
        "angry": "😠",
        "surprised": "😮",
        "cool": "😎",

        # Animals
        "dog": "🐶",
        "cat": "🐱",
        "mouse": "🐭",
        "hamster": "🐹",
        "rabbit": "🐰",
        "frog": "🐸",
        "tiger": "🐯",
        "bear": "🐻",
        "panda": "🐼",

        # Food and Drinks
        "food": "🍕",
        "apple": "🍎",
        "banana": "🍌",
        "cherries": "🍒",
        "strawberry": "🍓",
        "watermelon": "🍉",
        "burger": "🍔",
        "pizza": "🍕",
        "coffee": "☕",
        "cake": "🍰",

        # Nature and Weather
        "sun": "☀️",
        "moon": "🌙",
        "star": "⭐",
        "cloud": "☁️",
        "rain": "🌧️",
        "snow": "❄️",
        "thunder": "⚡",
        "leaf": "🍃",
        "flower": "🌸",

        # Objects
        "heart": "❤️",
        "gift": "🎁",
        "balloon": "🎈",
        "phone": "📱",
        "computer": "💻",
        "camera": "📷",
        "watch": "⌚",
        "key": "🔑",

        # Activities
        "soccer": "⚽",
        "basketball": "🏀",
        "baseball": "⚾",
        "guitar": "🎸",
        "microphone": "🎤",
        "gamepad": "🎮",
    }

    if emoji_name not in emoji_map:
        print(f"Emoji '{emoji_name}' not recognized. Using Gift")
        emoji_name = "gift" #Default to gift

    emoji_char = emoji_map.get(emoji_name)
    if not emoji_char:
        emoji_char = emoji.emojize(f":{emoji_name}:")

    emoji_char = emoji.emojize(f":{emoji_name}:")

    # Create a new 40x40 white image.
    img_size = (40, 40)
    image = Image.new("RGBA", img_size, "white")
    draw = ImageDraw.Draw(image)

    # Load a font that supports emojis. On some systems, you might need to supply a TTF file path.
    # We try the default PIL font as a fallback, but note: it might not support color emoji.
    font = None
    try:
        # Attempt to load a common emoji-supporting font if available.
        # For example, on Windows you might have "seguiemj.ttf" (Segoe UI Emoji) in C:\Windows\Fonts.
        emoji_font_path = None
        if os.name == "nt":
            potential_path = r"C:\Windows\Fonts\seguiemj.ttf"
            if os.path.exists(potential_path):
                emoji_font_path = potential_path
        else:
            # On many Linux systems, you can try "NotoColorEmoji.ttf"
            potential_path = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
            if os.path.exists(potential_path):
                emoji_font_path = potential_path

        if emoji_font_path:
            font = ImageFont.truetype(emoji_font_path, 32)
        else:
            # Fallback to the default font (which might not render emoji properly)
            font = ImageFont.load_default()
    except Exception as e:
        font = ImageFont.load_default()

    # Get text size for centering the emoji
    bbox = draw.textbbox((0, 0), emoji_char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position to center the emoji in the image.
    x = (img_size[0] - text_width) / 2
    y = (img_size[1] - text_height) / 2

    # Draw the emoji
    draw.text((x, y), emoji_char, font=font, fill="black")

    return image

# scrub strings for lookups
def to_upper_and_remove_spaces(input_string):
    # Convert to uppercase
    upper_string = input_string.upper()
    # Remove spaces
    result_string = upper_string.replace(" ", "")
    return result_string




# Create a 2D array for the world or load from CSV
def generate_world(grid_size):
    world = [['grass' for _ in range(grid_size)] for _ in range(grid_size)]
    for row in range(grid_size):
        for col in range(grid_size):
            rand = random.random()
            if rand < 0.1:
                world[row][col] = 'rock'
            elif rand < 0.3:
                world[row][col] = 'water'
    return world


def load_world_from_csv(filename):
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        return [row for row in reader]


def save_world_to_csv(filename, world, players):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in world:
            writer.writerow(row)
        # Save player positions, names, bios, tasks, and inventory
        for player in players:
            # Adding inventory to the saved data
            inventory_item = player.inventory if player.inventory is not None else "None"
            writer.writerow(['player', player.name, player.x, player.y, player.stats['Health'],
                             player.stats['Speed'], player.stats['Strength'], player.bio,
                             player.tasks, inventory_item])
    
    # Now also save to history.json
    history_data = []
    if os.path.exists('history.json'):
        try:
            with open('history.json', 'r') as f:
                history_data = json.load(f)
        except json.JSONDecodeError:
            history_data = []
    
    # Create new state entry
    current_state = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': []
    }
    
    # Add world data
    for row in world:
        current_state['data'].append(row)
    
    # Add player data
    for player in players:
        inventory_item = player.inventory if player.inventory is not None else "None"
        current_state['data'].append(['player', player.name, player.x, player.y, player.stats['Health'],
                                    player.stats['Speed'], player.stats['Strength'], player.bio,
                                    player.tasks, inventory_item])
    
    # Append to history
    history_data.append(current_state)
    
    # Save updated history
    with open('history.json', 'w') as f:
        json.dump(history_data, f, indent=2)


def generate_unique_position(existing_positions):
    while True:
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if (x, y) not in existing_positions:
            return x, y


conversation_dir = 'conversations'
os.makedirs(conversation_dir, exist_ok=True)

# Global conversation log
global_conversation_log = []


def save_conversation_history(player):
    filename = os.path.join(conversation_dir, f"{player.name.replace(' ', '_')}_conversation.txt")
    with open(filename, 'w') as file:
        file.write("\n".join(player.all_responses))


def load_conversation_history(player):
    filename = os.path.join(conversation_dir, f"{player.name.replace(' ', '_')}_conversation.txt")
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            player.all_responses = file.read().splitlines()


def save_global_conversations():
    filename = os.path.join(conversation_dir, 'global_conversation.txt')

    # Make sure the directory exists
    os.makedirs(conversation_dir, exist_ok=True)

    # Open file in append mode to create if not exists and append if it does
    with open(filename, 'a') as file:
        for entry in global_conversation_log:
            file.write(entry + "\n")


import openai

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def llm_request(conversation_list, request_string):
    global is_chatting
    is_chatting = True

    # Before making a blocking call, update display to show chat status.
    draw_chatting_notification(screen)
    pygame.display.flip()

    # Combine the conversation history with the request string.
    conversation_history = "\n".join(conversation_list)
    # Construct the full prompt
    prompt = f"{conversation_history}\n{request_string}"
    try:
        # Make an API call using the chat-based endpoint
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use the appropriate model
            messages=[
                {"role": "system", "content": "You are a human conversing."},
                {"role": "user", "content": prompt}
            ],
        )
        # Extract and return the response text from the message
        response_message = response.choices[0].message.content
        is_chatting = False
        return response_message

    except Exception as e:
        is_chatting = False
        # Handle errors appropriately
        return f"An error occurred: {str(e)}"

# Define tools that allow for map interactions
@tool("pickup_map_item")
def pickup_map_item(q:str) -> str:
    """
    Allow current player to pickup an item in place.
    
    Args:
        q (str): The name of the player who wants to pick up the item.
        
    Returns:
        str: A message indicating what item was picked up, or None if pickup failed.
    """
    print(f"\n\n\n\n------------------------------------- {q}")
    found_player = None
    lookup_player = to_upper_and_remove_spaces(q)
    if (player_lookup.get(lookup_player)):
        found_player = player_lookup.get(lookup_player)
    if not found_player:
        print(f"\n\n\n\n------------------------------------- Could not find Player named {q}")
        return
    else:
        print(f"\n\n\n\n------------------------------------- PLAYER PICKED UP ITEM {q}")
        item = found_player.pick_up_inplace()
        return f"picked up the item {item}"

@tool("drop_map_item")
def drop_map_item(q:str) -> str:
    """
    Allow current player to drop an item in place.
    
    Args:
        q (str): The name of the player who wants to drop the item.
        
    Returns:
        str: A message indicating what item was dropped, or None if drop failed.
    """
    found_player = None
    lookup_player = to_upper_and_remove_spaces(q)
    if (player_lookup.get(lookup_player)):
        found_player = player_lookup.get(lookup_player)
    if not found_player:
        print(f"\n\n\n\n------------------------------------- Could not find Player named {q}")
        return
    else:
        print(f"\n\n\n\n------------------------------------- PLAYER DROPPED ITEM {q}")
        item = found_player.place_inplace()
        return f"dropped the item {item}"


def get_neighboring_squares(player, world_map):
    """Get all neighboring squares including next nearest neighbors."""
    neighbors = []
    # Check all squares in a 3x3 grid centered on the player
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            new_x = player.x + dx
            new_y = player.y + dy
            # Skip if out of bounds
            if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                # Skip the center square (player's current position)
                if dx == 0 and dy == 0:
                    continue
                # Add direction and content
                direction = ""
                if dx == -1: direction += "North"
                if dx == 1: direction += "South"
                if dy == -1: direction += "West"
                if dy == 1: direction += "East"
                if not direction: direction = "Center"
                
                # Check if square is occupied by another player
                occupied_by = None
                for p in players:
                    if p.x == new_x and p.y == new_y:
                        occupied_by = p.name
                        break
                
                neighbors.append({
                    "direction": direction,
                    "content": world_map[new_x][new_y],
                    "occupied_by": occupied_by,
                    "dx": dx,
                    "dy": dy
                })
    return neighbors

@tool("decide_movement")
def decide_movement(q: str) -> str:
    """
    Decide which direction to move based on neighboring squares.
    
    Args:
        q (str): The name of the player who wants to move.
        
    Returns:
        str: A message indicating the movement decision.
    """
    found_player = None
    lookup_player = to_upper_and_remove_spaces(q)
    if player_lookup.get(lookup_player):
        found_player = player_lookup.get(lookup_player)
    if not found_player:
        return f"Could not find Player named {q}"
    
    neighbors = get_neighboring_squares(found_player, found_player.world_obj)
    return json.dumps({"neighbors": neighbors, "current_task": found_player.tasks})

def llm_request_crewai(
        conversation_list,
        request_string,
        player_obj,
):
    global is_chatting
    is_chatting = True
    # Before making a blocking call, update display to show chat status.
    draw_chatting_notification(screen)
    pygame.display.flip()



    # Combine the conversation history with the request string.
    conversation_history = "\n".join(conversation_list)
    # Construct the full prompt
    prompt = f"\n{request_string}"
    messages = [
        {"role": "user", "content": conversation_history},
        {"role": "system", "content": "You are a human conversing."},
        {"role": "user", "content": prompt}
    ]
    tasks = str(player_obj.tasks)
    my_bio = str(player_obj.bio)
    current_player_name = str(player_obj.name)

    llm = LLM(model="gpt-4o", temperature=0.7, api_key=os.environ["OPENAI_API_KEY"])

    conversation_agent = Agent(
        role="ConversationAgent",
        goal=f"I am a human who would like to converse with others",
        backstory=f"{prompt} in the past I have discussed {conversation_history[:100]}",
        llm=llm,
        memory=True,
        verbose=True,
    )

    conversation_task = Task(
        description="Start player conversation",
        expected_output="A detailed response if needed, or a summary response if needed, using the history of the user.",
        agent=conversation_agent,
    )

    ontask_agent = Agent(
        role="OntaskAgent",
        goal=f"stay on task, {prompt}",
        backstory=f"{prompt}, In the recent past I have discussed {conversation_history[:2]}, if I am not on task, "
                  f"discuss.",
        llm=llm,
        memory=True,
        verbose=True,
    )

    ontask_task = Task(
        description="Discuss something along the lines of the task",
        expected_output="A response using the task of the user and the conversation so far.",
        agent=ontask_agent,
    )

    map_pickup_interaction_agent = Agent(
        role="interact with the map",
        goal=f"decide whether or not to pickup the current map item based on my goal, if I want to pick it up say my name",
        backstory=f"{prompt}, to accomplish this, is the current map item something I should pickup and place "
                  f"Your name is {current_player_name} ",
        llm=llm,
        memory=True,
        verbose=True,
        context=[
            f"Role: Map interaction agent",
            f"Name: {current_player_name}",
            f"Tool Usage: Use pickup_map_item tool with argument '{current_player_name}'"
        ]
    )

    map_pickup_interaction_task = Task(
        description=f"""
        Decide whether to pickup the current map item. You are {current_player_name}.
        If you want to pick up the item, use the pickup_map_item tool with your name as the argument.
        Context: {prompt}
        """,
        expected_output="A decision about picking up the item and the result of using the tool if appropriate.",
        tools=[pickup_map_item],
        agent=map_pickup_interaction_agent
    )

    map_drop_interaction_agent = Agent(
        role="interact with the map",
        goal=f"decide whether or not to drop the current map item, if I want to drop it say my name",
        backstory=f"{prompt}, to accomplish this, is the current map item something I should drop and place. "
                  f"Your name is {current_player_name} ",
        llm=llm,
        memory=True,
        verbose=True,
        context=[
            f"Role: Map interaction agent",
            f"Name: {current_player_name}",
            f"Tool Usage: Use drop_map_item tool with argument '{current_player_name}'"
        ]
    )

    map_drop_interaction_task = Task(
        description=f"""
        Decide whether to drop the current inventory item. You are {current_player_name}.
        If you want to drop the item, use the drop_map_item tool with your name as the argument.
        Context: {prompt}
        """,
        expected_output="A decision about dropping the item and the result of using the tool if appropriate.",
        tools=[drop_map_item],
        agent=map_drop_interaction_agent
    )

    completedtask_agent = Agent(
        role="CompletedTaskAgent",
        goal=f"if my task completed {tasks}, say that is completed",
        backstory=f"{prompt}, In the recent past I have discussed {conversation_history[:6]}. If I am finished with my "
                  f"task mark the task as completed.",
        llm=llm,
        memory=True,
        verbose=True,
    )

    completedtask_task = Task(
        description="Mark the task is finished",
        expected_output=f"I have finished my task {tasks}",
        agent=completedtask_agent,
    )

    createtask_agent = Agent(
        role="CompletedTaskAgent",
        goal=f"If I have not completed my task {tasks}, respond with my task; otherwise, " 
             f"If my task completed {tasks}, generate a new task.",
        backstory=f"{prompt}. If I am finished with a "
                  f"task create a new 5 word task based on {my_bio}.",
        llm=llm,
        memory=True,
        verbose=True,
    )

    createtask_task = Task(
        description="Generate a new task",
        expected_output=f"I have finished my task {tasks}, generate a new one in plaintext with no commas.",
        agent=createtask_agent,
    )

    collaborate_agent = Agent(
        role="OntaskAgent",
        goal=f"help with the task",
        backstory=f"{prompt}, I am capable of helping any task..",
        llm=llm,
        memory=True,
        verbose=True,
    )

    collaborate_task = Task(
        description="Help with a task",
        expected_output="A short response to help complete the task.",
        agent=collaborate_agent,
    )

    uniqueness_agent = Agent(
        role="UniquenessAgent",
        goal=f"I need to be less repetitive in my responses",
        backstory=f"{prompt}, rephrase the response if I find myself repeating the following  {conversation_history[:100]}",
        llm=llm,
        memory=True,
        verbose=True,
    )

    uniqueness_task = Task(
        description="Start player conversation",
        expected_output="A unique response to the question using the history of the user in plaintext with no commas",
        agent=uniqueness_agent,
    )

    try:
        ## Conversation
        crew = Crew(
            agents=[ontask_agent, collaborate_agent, uniqueness_agent, conversation_agent, map_pickup_interaction_agent,
                    map_drop_interaction_agent],
            tasks=[ontask_task, collaborate_task, uniqueness_task, conversation_task, map_pickup_interaction_task, map_drop_interaction_task,
                   ],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff(inputs={
            'prompt': prompt,
            'history': conversation_history,
            'messages': messages,
            'tasks': str(player_obj.tasks),
            'my_bio': str(player_obj.bio),
            'current_player_name': str(player_obj.name),
        })

        ## Task reflection
        taskCrew = Crew(
            agents=[completedtask_agent, createtask_agent],
            tasks=[completedtask_task, createtask_task],
            process=Process.sequential,
            verbose=True
        )

        taskResult = taskCrew.kickoff(inputs={
            'prompt': prompt,
            'history': conversation_history,
            'messages': messages,
            'tasks': str(player_obj.tasks),
            'my_bio': str(player_obj.bio),
            'current_player_name': str(player_obj.name),
        })


        # let's accept task reassignment 30% of the time
        if random.random() < 0.3:
            old_task = str(player_obj.tasks)
            player_obj.tasks = str(taskResult)
            global_conversation_log.append(f"{player_obj.name} (changing tasks): completed({old_task}), started({player_obj.tasks})")

        is_chatting = False

        return str(result.raw)

    except Exception as e:
        is_chatting = False
        return f"An error occurred: {str(e)}"



def wrap_text(text, font, max_width):
    """Wraps the text to fit within the specified width."""
    words = text.split(' ')
    wrapped_lines = []
    current_line = ""

    for word in words:
        # Check the width of the new line if we add this word
        if font.size(current_line + word)[0] <= max_width:
            current_line += word + " "
        else:
            wrapped_lines.append(current_line)
            current_line = word + " "

    if current_line:
        wrapped_lines.append(current_line)

    return wrapped_lines


# Button class for handling the Pause/Unpause button
class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = (173, 216, 230)  # Light blue
        self.outline_color = (0, 0, 0)  # Black for outline
        self.font = pygame.font.Font(None, 36)

    def draw(self, screen):
        # Draw the outline
        pygame.draw.rect(screen, self.outline_color, self.rect, 2)  # 2 pixels border
        # Draw the inside of the button
        pygame.draw.rect(screen, self.color, self.rect.inflate(-4, -4))
        # Render the text
        text_surf = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)


# Button setup
button_width, button_height = 120, 40
pause_button = Button(WIDTH - button_width - 10, HEIGHT - button_height - 10, button_width, button_height, "Pause")
autonomy_button = Button(WIDTH - button_width - 10, HEIGHT - (button_height * 2) - 20, button_width, button_height, "autonomy")


class Player:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
        self.name = fake.name()
        self.stats = {'Health': 100, 'Speed': 5, 'Strength': 10}
        self.bio = fake.text(max_nb_chars=100)
        self.tasks = fake.text(max_nb_chars=100)
        self.current_conversation = ""
        self.all_responses = []
        self.health = 100
        self.speed = 100
        self.strength = 100
        self.current_scroll = 0
        self.inventory = None
        self.world_obj = None
        load_conversation_history(self)

    def pick_up(self, x, y, world_map):
        self.inventory = world_map[y][x]
        world_map[y][x] = 'rock'

    def pick_up_inplace(self):
        if self.world_obj is None:
            print(f"Error: {self.name} has no reference to world object")
            return None
        if self.inventory is not None:
            print(f"Error: {self.name} is already holding {self.inventory}")
            return None
        try:
            current_item = self.world_obj[self.x][self.y]
            if current_item == 'grass' or current_item == 'rock':
                print(f"Error: {self.name} cannot pick up {current_item}")
                return None
            self.inventory = current_item
            self.world_obj[self.x][self.y] = 'grass'
            print(f"Success: {self.name} picked up {current_item}")
            return current_item
        except Exception as e:
            print(f"Error during pickup: {str(e)}")
            return None

    def place(self, x, y, world_map):
        if self.inventory is not None:
            world_map[y][x] = self.inventory
            self.inventory = None

    def place_inplace(self):
        if self.world_obj is None:
            print(f"Error: {self.name} has no reference to world object")
            return None
        if self.inventory is None:
            print(f"Error: {self.name} has no item to place")
            return None
        try:
            current_spot = self.world_obj[self.x][self.y]  # Fixed coordinate order to match world.csv layout
            if current_spot != 'grass':
                print(f"Error: {self.name} cannot place item on {current_spot}")
                return None
            item_to_place = self.inventory
            self.world_obj[self.x][self.y] = self.inventory  # Fixed coordinate order to match world.csv layout
            self.inventory = None
            print(f"Success: {self.name} placed {item_to_place}")
            return item_to_place
        except Exception as e:
            print(f"Error during placement: {str(e)}")
            return None


    def who_am_i_string(self):
        return f"I am {self.name}.My background is: {self.bio}.I am currently {self.tasks}."

    def generate_conversation(self):
        self.current_conversation = self.who_am_i_string()
        response_string = llm_request_crewai(self.all_responses, self.current_conversation, self)
        self.all_responses.append(response_string)
        return response_string

    def respond_conversation(self, text, other_player_name):
        self.current_conversation = self.who_am_i_string()
        self.current_conversation += f"I am talking to someone who is saying: {text}"
        self.current_conversation += "In one sentence, what should I respond with"
        response_string = llm_request_crewai(self.all_responses, self.current_conversation, self)
        self.all_responses.append(response_string)
        print(f"{self.name} convo:{self.current_conversation}")
        print(f"{self.name} response:{response_string}")

        # Add to global conversation log
        global_conversation_log.append(f"{other_player_name} (talking to {self.name}): {text}")
        global_conversation_log.append(f"{self.name} (talking to {other_player_name}): {response_string}")

        save_conversation_history(self)
        return response_string


def find_nearby_player(player, players):
    for other_player in players:
        if other_player != player:
            if abs(player.x - other_player.x) <= 1 and abs(player.y - other_player.y) <= 1:
                return other_player
    return None


# Check if world.csv exists
world_filename = 'world.csv'
players = []
player_lookup = {}

if os.path.exists(world_filename):
    data = load_world_from_csv(world_filename)
    world = [row for row in data if row[0] not in ['player']]
    players_data = [row for row in data if row[0] == 'player']

    for player_data in players_data:
        _, name, x, y, health, speed, strength, bio, tasks, inventory_item = player_data
        player = Player(int(x), int(y), player_imgs[len(players)])
        player.name = name
        player.stats = {'Health': int(health), 'Speed': int(speed), 'Strength': int(strength)}
        player.bio = bio
        player.tasks = tasks
        player.world_obj = world
        player.inventory = inventory_item if inventory_item != "None" else None
        load_conversation_history(player)
        players.append(player)
        player_lookup[to_upper_and_remove_spaces(player.name)] = player

    print(f"\n\n{players_data}")
    print(f"\n\n{player_lookup}")

else:
    world = generate_world(GRID_SIZE)
    # Initialize players with unique positions
    existing_positions = set()
    # Use a default of 3 players when creating a new world
    num_new_players = 3
    for i in range(num_new_players):
        x, y = generate_unique_position(existing_positions)
        existing_positions.add((x, y))
        player = Player(x, y, player_imgs[i])
        player.world_obj = world
        players.append(player)

    save_world_to_csv(world_filename, world, players)

# Set active_player_idx based on actual number of players
num_players = len(players)
if num_players == 0:
    print("Error: No players found in world.csv")
    pygame.quit()
    exit()
active_player_idx = random.randint(0, num_players - 1)


def handle_map_interactions(player):
    """Handle map interactions after a player moves to a new position."""
    # Get surrounding squares information
    neighbors = get_neighboring_squares(player, player.world_obj)
    
    # Create map observation entry
    map_observation = f"MAP: At ({player.x}, {player.y}). "
    if player.inventory:
        map_observation += f"Holding {player.inventory}. "
    current_tile = player.world_obj[player.x][player.y]
    map_observation += f"Current tile: {current_tile}. Surroundings: "
    
    # Add information about each neighboring square
    for neighbor in neighbors:
        map_observation += f"{neighbor['direction']}: {neighbor['content']}"
        if neighbor['occupied_by']:
            map_observation += f"(occupied by {neighbor['occupied_by']})"
        map_observation += ", "
    
    # Add the map observation to player's conversation history
    player.all_responses.append(map_observation)
    
    # Create prompt for CrewAI
    prompt = f"{player.who_am_i_string()} {map_observation}"

    # Use CrewAI to make decisions about picking up or dropping items
    response = llm_request_crewai(
        player.all_responses,
        prompt,
        player
    )
    
    # Add the interaction result to the player's responses
    player.all_responses.append(response)
    return response

def move_player(player, dx, dy):
    new_x, new_y = player.x + dx, player.y + dy
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and world[new_x][new_y] != 'rock':
        # Check if the new position is occupied by another player
        if not any(p.x == new_x and p.y == new_y for p in players):
            old_x, old_y = player.x, player.y
            player.x, player.y = new_x, new_y
            
            # Only trigger map interactions if the player actually moved
            if old_x != new_x or old_y != new_y:
                handle_map_interactions(player)
                return True  # Indicate successful movement
    return False  # Indicate failed movement

def decide_next_move(player):
    # Use CrewAI to decide the next move for a player.
    llm = LLM(model="gpt-4o", temperature=0.7, api_key=os.environ["OPENAI_API_KEY"])

    # Get recent map history
    map_history = []
    for response in player.all_responses[-50:]:  # Look at last 50 responses
        if response.startswith("MAP:"):
            map_history.append(response)

    movement_agent = Agent(
        role="MovementAgent",
        goal=f"decide where to move next based on surroundings and current task",
        backstory=f"I analyze the surroundings and make movement decisions based on the current task: {player.tasks}. "
                 f"I must return movement decisions in JSON format with dx and dy values. "
                 f"Recent map history:\n" + "\n".join(map_history),
        llm=llm,
        memory=True,
        verbose=True,
        tools=[decide_movement]
    )

    movement_task = Task(
        description=(
            f"Analyze surroundings and decide where to move. You are {player.name}.\n"
            f"Use the decide_movement tool with your name as the argument to get information about neighboring squares.\n"
            f"Based on the response and your map history, determine the best direction to move considering your current task.\n\n"
            f"Map History (use this to make informed decisions about where to explore or return to):\n"
            f"{chr(10).join(map_history)}\n\n"
            f"Context: Moving towards task completion: {player.tasks}\n\n"
            "IMPORTANT: You must format your final answer as a JSON string with the following structure:\n"
            "{\n"
            '    "dx": <number between -1 and 1>,\n'
            '    "dy": <number between -1 and 1>,\n'
            '    "reason": "explanation for the movement"\n'
            "}\n\n"
            "For example:\n"
            "{\n"
            '    "dx": -1,\n'
            '    "dy": 1,\n'
            '    "reason": "Moving northeast towards the target location"\n'
            "}"
        ),
        expected_output="A JSON string containing dx, dy values and a reason for the movement.",
        tools=[decide_movement],
        agent=movement_agent
    )

    crew = Crew(
        agents=[movement_agent],
        tasks=[movement_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    
    # Parse the movement decision and execute it
    try:
        # Try to find a JSON string in the response
        response_text = result.raw
        # Look for JSON-like structure in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx + 1]
            decision = json.loads(json_str)
            
            if "dx" in decision and "dy" in decision:
                dx = int(decision["dx"])
                dy = int(decision["dy"])
                # Ensure we only move one unit at a time
                dx = max(min(dx, 1), -1)
                dy = max(min(dy, 1), -1)
                if "reason" in decision:
                    print(f"Moving {player.name}: {decision['reason']}")
                return move_player(player, dx, dy)
        raise ValueError("No valid movement decision found in response")
    except Exception as e:
        print(f"Error processing movement decision: {str(e)}")
        print(f"Raw response: {response_text}")
        return False

def move_away(player, other_player):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    random.shuffle(directions)
    for dx, dy in directions:
        new_x, new_y = player.x + dx, player.y + dy
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and world[new_x][new_y] != 'rock':
            if not any(p.x == new_x and p.y == new_y for p in players):
                player.x, player.y = new_x, new_y
                break


def move_inactive_players(players, active_idx):
    for i, player in enumerate(players):
        if i != active_idx:
            nearby_player = find_nearby_player(player, players)
            if nearby_player:
                conversation = player.generate_conversation()
                response = nearby_player.respond_conversation(conversation, player.name)
                print(f"{player.name}: Converation {conversation}")
                print(f"{nearby_player.name}: Response {response}")
                player.all_responses.append(response)
                move_away(player, nearby_player)
                move_away(nearby_player, player)
            else:
                # Use the new AI-driven movement system
                decide_next_move(player)


def draw_world(surface, world, players):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            tile_type = world[row][col]
            if tile_type == 'grass':
                image = grass_img
            elif tile_type == 'rock':
                image = rock_img
            elif tile_type == 'water':
                image = water_img
            else:
                # For any other tile type, try to load its image or generate it
                image = load_image(f'{tile_type}.png', (CELL_SIZE, CELL_SIZE))
            surface.blit(image, (col * CELL_SIZE, row * CELL_SIZE))
    for player in players:
        surface.blit(player.image, (player.y * CELL_SIZE, player.x * CELL_SIZE))


# Update draw_stats to show recent response
def draw_stats(surface, player, x_offset, y_offset):
    global WRAPPED_LINES # For UI calculations
    font = pygame.font.Font(None, 36)
    y = y_offset

    # Display player name
    name_text = font.render(player.name, True, (0, 0, 0))
    surface.blit(name_text, (x_offset, y))
    y += 40


    # Static positions for Bio and Tasks
    bio_label_y = y
    tasks_label_y = y + 40

    # Render the labels
    bio_label = font.render('Bio:', True, (0, 0, 0))
    tasks_label = font.render('Tasks:', True, (0, 0, 0))

    # Display Bio and Tasks in the same section
    surface.blit(bio_label, (x_offset, bio_label_y))
    surface.blit(tasks_label, (x_offset, tasks_label_y))

    # Display recent response
    if player.all_responses:
        y+=80

        wrapped_lines = wrap_text('Recent: ' + player.all_responses[-1], font, WIDTH - x_offset - 20)
        WRAPPED_LINES = wrapped_lines
        # Scrolling debug
        # print(f"{player.current_scroll}, {len(wrapped_lines)} :  {player.current_scroll + (RESPONSE_BOX_HEIGHT // 20)} : {WRAPPED_LINES},")
        for line in WRAPPED_LINES[player.current_scroll: player.current_scroll + (RESPONSE_BOX_HEIGHT // 20)]:
            surface.blit(font.render(line, True, (0, 0, 0)), (x_offset, y))
            y += 20

        # Scroll indicator (optional)
        if len(wrapped_lines) > (RESPONSE_BOX_HEIGHT // 20) and paused:
            surface.blit(font.render("[Up/Down]:Scroll", True, (155, 155, 155)), (x_offset, y+10))


    # Draw input boxes in paused mode
    if paused:
        bio_box.rect.topleft = (x_offset + 80, bio_label_y)
        tasks_box.rect.topleft = (x_offset + 80, tasks_label_y)
        bio_box.draw(screen)
        tasks_box.draw(screen)
    else:
        # Unpaused, display current Bio and Tasks
        bio_text = font.render(player.bio, True, (0, 0, 0))
        tasks_text = font.render(player.tasks, True, (0, 0, 0))
        surface.blit(bio_text, (x_offset + 80, bio_label_y))
        surface.blit(tasks_text, (x_offset + 80, tasks_label_y))

# Function to draw "Chatting..." notification
def draw_chatting_notification(surface):
    notification_rect = pygame.Rect(WIDTH - 300, HEIGHT - 50, 150, 40)
    pygame.draw.rect(surface, (200, 200, 255), notification_rect)  # White box
    pygame.draw.rect(surface, (0, 0, 0), notification_rect, 2)  # Black border
    font = pygame.font.Font(None, 30)
    text_surf = font.render("Chatting...", True, (0, 0, 0))
    surface.blit(text_surf, (notification_rect.x + 10, notification_rect.y + 10))


class InputBox:
    def __init__(self, x, y, w, h, text=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_active = (173, 216, 230)  # Light blue
        self.color_inactive = (0, 0, 0)  # Black
        self.text = text
        self.txt_surface = pygame.font.Font(None, 36).render(text, True, self.color_inactive)
        self.active = False
        self.enter_pressed = False  # Track if Enter has been pressed

    def handle_event(self, event, paused):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle the active variable.
            if self.rect.collidepoint(event.pos):
                if paused:  # Only activate if the game is paused
                    self.active = not self.active
                else:
                    self.active = False
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    self.enter_pressed = True  # Mark Enter as pressed
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = pygame.font.Font(None, 36).render(self.text, True, self.color_active if self.active else self.color_inactive)

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color_active if self.active else self.color_inactive, self.rect, 2)

    def update_text(self, text):
        self.text = text
        self.txt_surface = pygame.font.Font(None, 36).render(text, True,
                                                             self.color_active if self.active else self.color_inactive)

    def reset_enter_pressed(self):
        self.enter_pressed = False
def get_player_at_pos(players, x, y):
    for i, player in enumerate(players):
        player_rect = pygame.Rect(player.y * CELL_SIZE, player.x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        if player_rect.collidepoint(x, y):
            return i
    return None

# Initial configuration
sub_epoch = 30  # time increments
idle_mod = 20 % sub_epoch  # update cycle for idle things
idle_count = 0
running = True
paused = True
is_autonomous = False  # Track if player is in autonomous mode
selected_player = players[active_player_idx]

# Create input boxes for player attributes
stat_boxes = [
    InputBox(GRID_SIZE * CELL_SIZE + 100, 100 + i * 40, 100, 32, str(getattr(selected_player, stat.lower())))
    for i, stat in enumerate(['Health', 'Speed', 'Strength'])
]
# 260, 350
bio_box = InputBox(GRID_SIZE * CELL_SIZE + 20, 20, 300, 32, selected_player.bio)
tasks_box = InputBox(GRID_SIZE * CELL_SIZE + 20, 50, 300, 32, selected_player.tasks)
#input_boxes = stat_boxes + [bio_box, tasks_box]
input_boxes = [bio_box, tasks_box]

def save_history(world, players):
    # Load existing history
    history_data = []
    if os.path.exists('history.json'):
        try:
            with open('history.json', 'r') as f:
                history_data = json.load(f)
        except json.JSONDecodeError:
            history_data = []
    
    # Create new state entry
    current_state = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': []
    }
    
    # Add world data
    for row in world:
        current_state['data'].append(row)
    
    # Add player data
    for player in players:
        inventory_item = player.inventory if player.inventory is not None else "None"
        current_state['data'].append(['player', player.name, player.x, player.y, player.stats['Health'],
                                    player.stats['Speed'], player.stats['Strength'], player.bio,
                                    player.tasks, inventory_item])
    
    # Append to history
    history_data.append(current_state)
    
    # Save updated history
    with open('history.json', 'w') as f:
        json.dump(history_data, f, indent=2)

while running:
    idle_count = (idle_count + 1) % sub_epoch
    
    # Save history at the start of each loop iteration
    save_history(world, players)
    
    for event in pygame.event.get():
        if pause_button.is_clicked(event):
            paused = not paused
            pause_button.text = "Unpause" if paused else "Pause"
            save_world_to_csv(world_filename, world, players)
        if autonomy_button.is_clicked(event):
            is_autonomous = not is_autonomous
            autonomy_button.text = "user ctrl" if is_autonomous else "autonomy"
        if event.type == pygame.QUIT:
            save_world_to_csv(world_filename, world, players)
            for player in players:
                save_conversation_history(player)
            save_global_conversations()
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            clicked_player_idx = get_player_at_pos(players, mouse_x, mouse_y)
            if clicked_player_idx is not None:
                active_player_idx = clicked_player_idx
                selected_player = players[active_player_idx]
                for stat, box in zip(selected_player.stats.values(), stat_boxes):
                    box.update_text(str(stat))
                bio_box.update_text(selected_player.bio)
                tasks_box.update_text(selected_player.tasks)

        # Handle input boxes even if paused
        for box in input_boxes:
            box.handle_event(event, paused)

    # Process other game logic only if not paused
    if not paused:
        if is_autonomous:
            # Autonomous movement - move player based on their tasks
            if idle_count % idle_mod == 0:
                move_inactive_players([players[active_player_idx]], -1)  # -1 means no active player to avoid
        else:
            # Manual keyboard control
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                move_player(players[active_player_idx], -1, 0)
            if keys[pygame.K_DOWN]:
                move_player(players[active_player_idx], 1, 0)
            if keys[pygame.K_LEFT]:
                move_player(players[active_player_idx], 0, -1)
            if keys[pygame.K_RIGHT]:
                move_player(players[active_player_idx], 0, 1)

        if idle_count % idle_mod == 0:
            move_inactive_players(players, active_player_idx)

    else:  # When paused, allow scrolling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            if selected_player.current_scroll > 0:
                selected_player.current_scroll -= SCROLL_SPEED
        elif keys[pygame.K_DOWN]:
            if selected_player.current_scroll < len(WRAPPED_LINES) - (RESPONSE_BOX_HEIGHT // 20):
                selected_player.current_scroll += SCROLL_SPEED

    screen.fill((255, 255, 255))
    draw_world(screen, world, players)

    for stat, box in zip(selected_player.stats.keys(), stat_boxes):
        try:
            value = int(box.text)
            selected_player.stats[stat] = value
        except ValueError:
            pass
    selected_player.bio = bio_box.text
    selected_player.tasks = tasks_box.text

    draw_stats(screen, selected_player, GRID_SIZE * CELL_SIZE + 20, 10)

    # allow input updates if paused
    if paused:
        for box in input_boxes:
            box.draw(screen)

    # Draw the pause/unpause button
    pause_button.draw(screen)
    autonomy_button.draw(screen)

    # Draw chatting notification if chatting is enabled
    if is_chatting:
        draw_chatting_notification(screen)


    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()