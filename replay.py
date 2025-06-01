import pygame
import json
import csv
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from openai import OpenAI

# Initialize Pygame
pygame.init()

# Constants (copied from conversation_party_crewai.py)
GRID_SIZE = 10
CELL_SIZE = 40
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE * 2, GRID_SIZE * CELL_SIZE
FPS = 1  # 1 frame per second for replay
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40

# Display setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("World Replay")
clock = pygame.time.Clock()

# Button class (copied and modified from conversation_party_crewai.py)
class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = (173, 216, 230)  # Light blue
        self.outline_color = (0, 0, 0)  # Black for outline
        self.font = pygame.font.Font(None, 36)

    def draw(self, screen):
        pygame.draw.rect(screen, self.outline_color, self.rect, 2)
        pygame.draw.rect(screen, self.color, self.rect.inflate(-4, -4))
        text_surf = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

# Asset loading functions (copied from conversation_party_crewai.py)
def ensure_asset_exists(asset_name: str, assets_dir: str) -> str:
    if asset_name.startswith('assets/'):
        asset_name = asset_name[7:]
        
    asset_path = os.path.join(assets_dir, asset_name)
    os.makedirs(assets_dir, exist_ok=True)
    
    if os.path.exists(asset_path):
        return asset_path
        
    try:
        client = OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Create a simple pixel art style icon for {asset_name.split('.')[0]}. The image should be minimal and clear, suitable for a 31x39 pixel game asset.",
            size="1024x1024",
            n=1,
        )
        
        image_url = response.data[0].url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((31, 39), Image.Resampling.LANCZOS)
        img.save(asset_path)
        return asset_path
        
    except Exception as e:
        print(f"Failed to generate image for {asset_name}: {str(e)}")
        return None

def load_image(filename: str, size=(31,39)) -> pygame.Surface:
    try:
        if filename.startswith('assets/'):
            filename = filename[7:]
            
        asset_path = os.path.join(assets_dir, filename)
        os.makedirs(assets_dir, exist_ok=True)
        
        if os.path.exists(asset_path):
            image = pygame.image.load(asset_path)
            return pygame.transform.scale(image, size)
            
        try:
            client = OpenAI()
            response = client.images.generate(
                model="dall-e-3",
                prompt=f"Create a simple pixel art style icon for {filename.split('.')[0]}. The image should be minimal and clear, suitable for a {size[0]}x{size[1]} pixel game asset.",
                size="1024x1024",
                n=1,
            )
            
            image_url = response.data[0].url
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(asset_path)
            return pil_to_pygame(img)
            
        except Exception as e:
            print(f"Failed to generate image for {filename}: {str(e)}")
            surface = pygame.Surface(size)
            surface.fill((255, 0, 255))
            return surface
            
    except Exception as e:
        print(f"Error loading image {filename}: {str(e)}")
        surface = pygame.Surface(size)
        surface.fill((255, 0, 255))
        return surface

def pil_to_pygame(pil_image: Image.Image) -> pygame.Surface:
    pil_image = pil_image.convert("RGBA")
    mode = pil_image.mode
    size = pil_image.size
    data = pil_image.tobytes()
    return pygame.image.fromstring(data, size, mode)

# Load images
assets_dir = 'assets'
grass_img = load_image('grass.png', (CELL_SIZE, CELL_SIZE))
rock_img = load_image('rock.png', (CELL_SIZE, CELL_SIZE))
water_img = load_image('water.png', (CELL_SIZE, CELL_SIZE))
player_imgs = [load_image(f'player_{i}.png', (CELL_SIZE, CELL_SIZE)) for i in range(5)]

class Player:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
        self.name = ""
        self.stats = {'Health': 100, 'Speed': 5, 'Strength': 10}
        self.bio = ""
        self.tasks = ""
        self.inventory = None

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
                image = load_image(f'{tile_type}.png', (CELL_SIZE, CELL_SIZE))
            surface.blit(image, (col * CELL_SIZE, row * CELL_SIZE))
    for player in players:
        surface.blit(player.image, (player.y * CELL_SIZE, player.x * CELL_SIZE))

def load_world_state(data):
    world = [row for row in data if row[0] not in ['player']]
    players = []
    players_data = [row for row in data if row[0] == 'player']
    
    for player_data in players_data:
        _, name, x, y, health, speed, strength, bio, tasks, inventory = player_data
        player = Player(int(x), int(y), player_imgs[len(players)])
        player.name = name
        player.stats = {'Health': int(health), 'Speed': int(speed), 'Strength': int(strength)}
        player.bio = bio
        player.tasks = tasks
        player.inventory = None if inventory == "None" else inventory
        players.append(player)
    
    return world, players

def load_history():
    if not os.path.exists('history.json'):
        return []
    with open('history.json', 'r') as f:
        return json.load(f)

# Create buttons
right_panel_x = GRID_SIZE * CELL_SIZE + 20  # Start of right panel
button_spacing = 20
button_y_start = HEIGHT - (BUTTON_HEIGHT + button_spacing) * 4  # Start buttons from bottom, spacing for 4 buttons
play_button = Button(right_panel_x, button_y_start, BUTTON_WIDTH, BUTTON_HEIGHT, "Play")
pause_button = Button(right_panel_x, button_y_start + (BUTTON_HEIGHT + button_spacing), BUTTON_WIDTH, BUTTON_HEIGHT, "Pause")
prev_button = Button(right_panel_x, button_y_start + 2 * (BUTTON_HEIGHT + button_spacing), BUTTON_WIDTH, BUTTON_HEIGHT, "Previous")
next_button = Button(right_panel_x, button_y_start + 3 * (BUTTON_HEIGHT + button_spacing), BUTTON_WIDTH, BUTTON_HEIGHT, "Next")

# Main replay loop
def main():
    history = load_history()
    if not history:
        print("No history found. Please run the game first to generate history.")
        return

    current_frame = 0
    is_playing = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle button clicks
            if play_button.is_clicked(event):
                is_playing = True
            elif pause_button.is_clicked(event):
                is_playing = False
            elif prev_button.is_clicked(event):
                current_frame = max(0, current_frame - 1)
                is_playing = False
            elif next_button.is_clicked(event):
                current_frame = min(len(history) - 1, current_frame + 1)
                is_playing = False

        # Update frame if playing
        if is_playing and current_frame < len(history) - 1:
            current_frame += 1

        # Draw current frame
        screen.fill((255, 255, 255))
        
        # Load and draw current world state
        current_state = history[current_frame]
        world, players = load_world_state(current_state['data'])
        draw_world(screen, world, players)

        # Draw timestamp and frame counter in right panel
        font = pygame.font.Font(None, 36)
        
        # Draw title
        title_text = font.render("World Replay", True, (0, 0, 0))
        screen.blit(title_text, (right_panel_x, 20))
        
        # Draw timestamp
        timestamp_text = font.render("Time:", True, (0, 0, 0))
        screen.blit(timestamp_text, (right_panel_x, 60))
        timestamp_value = font.render(current_state['timestamp'], True, (0, 0, 0))
        screen.blit(timestamp_value, (right_panel_x, 90))

        # Draw frame counter
        frame_text = font.render(f"Frame: {current_frame + 1}/{len(history)}", True, (0, 0, 0))
        screen.blit(frame_text, (right_panel_x, 130))

        # Draw buttons
        play_button.draw(screen)
        pause_button.draw(screen)
        prev_button.draw(screen)
        next_button.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main() 