import pygame
import random
import os
import csv
from faker import Faker
import openai
from openai import OpenAI

# Uncomment below if you want to set an openai key, or if it's not set in the environment
# os.environ["OPENAI_API_KEY"] = "ADD KEY HERE"

# Initialize Pygame and Faker
pygame.init()
fake = Faker()

# Screen dimensions and other constants
GRID_SIZE = 10
CELL_SIZE = 40
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE * 2, GRID_SIZE * CELL_SIZE
FPS = 30
NUM_PLAYERS = 3  # Number of players in the game
SCROLL_SPEED = 5
RESPONSE_BOX_HEIGHT = 100


# Display setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Conversation Party")
clock = pygame.time.Clock()
WRAPPED_LINES = ""
# Global variable to track chatting state
is_chatting = False


# Function to load and resize images
def load_image(filename, size):
    image = pygame.image.load(filename)
    return pygame.transform.scale(image, size)


# Load and resize images
assets_dir = 'assets'
grass_img = load_image(os.path.join(assets_dir, 'grass.png'), (CELL_SIZE, CELL_SIZE))
rock_img = load_image(os.path.join(assets_dir, 'rock.png'), (CELL_SIZE, CELL_SIZE))
water_img = load_image(os.path.join(assets_dir, 'water.png'), (CELL_SIZE, CELL_SIZE))
player_imgs = [load_image(os.path.join(assets_dir, f'player_{i}.png'), (CELL_SIZE, CELL_SIZE)) for i in range(5)]


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
        # Save player positions, names, bios, and tasks
        for player in players:
            writer.writerow(['player', player.name, player.x, player.y, player.stats['Health'], player.stats['Speed'],
                             player.stats['Strength'], player.bio, player.tasks])


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
    with open(filename, 'w') as file:
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
        load_conversation_history(self)

    def who_am_i_string(self):
        return f"I am {self.name}.My background is: {self.bio}.I am currently {self.tasks}."

    def generate_conversation(self):
        self.current_conversation = self.who_am_i_string()
        response_string = llm_request(self.all_responses, self.current_conversation)
        self.all_responses.append(response_string)
        return response_string

    def respond_conversation(self, text, other_player_name):
        self.current_conversation = self.who_am_i_string()
        self.current_conversation += f"I am talking to someone who is saying: {text}"
        self.current_conversation += "In one sentence, what should I respond with"
        response_string = llm_request(self.all_responses, self.current_conversation)
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

if os.path.exists(world_filename):
    data = load_world_from_csv(world_filename)
    world = [row for row in data if row[0] not in ['player']]
    players_data = [row for row in data if row[0] == 'player']

    players = []
    for player_data in players_data:
        _, name, x, y, health, speed, strength, bio, tasks = player_data
        player = Player(int(x), int(y), player_imgs[len(players)])
        player.name = name
        player.stats = {'Health': int(health), 'Speed': int(speed), 'Strength': int(strength)}
        player.bio = bio
        player.tasks = tasks
        load_conversation_history(player)
        players.append(player)
else:
    world = generate_world(GRID_SIZE)
    # Initialize players with unique positions
    players = []
    existing_positions = set()
    for i in range(NUM_PLAYERS):
        x, y = generate_unique_position(existing_positions)
        existing_positions.add((x, y))
        players.append(Player(x, y, player_imgs[i]))

    save_world_to_csv(world_filename, world, players)

active_player_idx = random.randint(0, NUM_PLAYERS - 1)


def move_player(player, dx, dy):
    new_x, new_y = player.x + dx, player.y + dy
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and world[new_x][new_y] != 'rock':
        # Check if the new position is occupied by another player
        if not any(p.x == new_x and p.y == new_y for p in players):
            player.x, player.y = new_x, new_y


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
                dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
                move_player(player, dx, dy)


def draw_world(surface, world, players):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            image = grass_img if world[row][col] == 'grass' else rock_img if world[row][col] == 'rock' else water_img
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
paused = False
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

while running:
    idle_count = (idle_count + 1) % sub_epoch
    for event in pygame.event.get():
        if pause_button.is_clicked(event):
            paused = not paused
            pause_button.text = "Unpause" if paused else "Pause"
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

    # Draw chatting notification if chatting is enabled
    if is_chatting:
        draw_chatting_notification(screen)


    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()