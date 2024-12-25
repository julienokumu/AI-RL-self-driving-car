import neat # NeuroEvolution of Augmenting Topologies
import pygame # game dev
import os # interaction with os
import math # mathematical calculations
import random # gen random numbers
import sys # system functions

# width and height of game window
width = 870
height = 515

# width and height of car
car_size_x = 30
car_size_y = 30

# color of border
border_color = (255, 255, 255, 255)

# counter for initial number of generations
current_generation = 0

# car class
class Car:
    def __init__(self):
        # load car image and scale it to the car size
        self.sprite = pygame.image.load('car.png').convert() # converted for faster blitting
        self.sprite = pygame.transform.scale(self.sprite, (car_size_x, car_size_y)) # scale the car image
        self.rotated_sprite = self.sprite # initialize rotated sprite
        self.position = [371, 415] # starting position of car
        self.angle = 0 # initial angle of car
        self.speed = 0 # initial speed of car
        self.speed_set = False # flag to set default speed later
        self.center = [self.position[0] + car_size_x / 2, self.position[1] + car_size_y / 2]
        self.radars = [] # list to store radar data
        self.drawing_radars = [] # list to store radars to be drawn
        self.alive = True # boolean to check if car is still alive
        self.distance = 0 # distance driven by driver
        self.time = 0 # time passed
    
    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position) # draw car on screen
        self.draw_radar(screen) # draw radars
    
    def draw_radar(self, screen):
        # draw all the radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1) # draw radar line
            pygame.draw.circle(screen, (0, 255, 0), position, 5) # draw radar circle

    def check_collision(self, game_map):
        self.alive = True # assume the car is alive
        for point in self.corners:
            # check if any corner of the car touches the border color
            if game_map.get_at((int(point[0]), int(point[1]))) == border_color:
                self.alive = False # car is not alive if it touches the border
                break
    
    def check_radar(self, degree, game_map):
        length = 0 # intialize radar length
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        # extends the radar until it hits the border or reahces a maximum length
        while not game_map.get_at((x, y)) == border_color and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        # calculate distance to the border and append to the radars list
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        # set the speed to 20 for the first time
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True
        # get rotated sprite and move the car in the right direction
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20) # prevent car from going too far left
        self.position[0] = min(self.position[0], width - 120) # prevent car from going too far down
        # increase distance and time
        self.distance += self.speed
        self.time += 1
        # move the car in the y-direction
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20) # prevent car from going too far up
        self.position[1] = min(self.position[1], width - 120) # prevent car from going to far down
        # new center of car
        self.center = [int(self.position[0]) + car_size_x / 2, int(self.position[1]) + car_size_y / 2]
        # calculate the four corners of the car
        length = 0.5 * car_size_x
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]
        # check for collisions and clear radars
        self.check_collision(game_map)
        self.radars.clear()
        # check radars from -90 to 120 degrees with a step size fo 45
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)
    
    def get_data(self):
        # get distance to the border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30) # normalize the distance
        return return_values
    
    def is_alive(self):
        # check if the car is still alive
        return self.alive
    
    def get_reward(self):
        # calculate the reward based on the distance driven
        return self.distance / (car_size_x / 2)
    
    def rotate_center(self, image, angle):
        # rotate the image around its center
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
    
def run_simulation(genomes, config):
    # empty collections for neural networks and cars
    nets = []
    cars = []
    # intialize pygame and display
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("AI F1 Car with RL by Julien Okumu")
    # for each genome, create a new neural network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0 # initalize fitness
        cars.append(Car()) # create a new car
    # clock settings
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30) 
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load('map.png').convert() # load game map
    global current_generation
    current_generation += 1 # increment generation counter
    counter = 0 # simple counter to roughly limit time
    while True:
        # exit on quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0) # exit program
                running = False
        # for each car, get the action it takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data()) # get output from the neural network
            choice = output.index(max(output)) # get index of the highest output
            if choice == 0:
                car.angle += 10 # turn left
            elif choice == 1:
                car.angle -= 10 # turn right
            elif choice == 2:
                if car.speed - 2 >= 12:
                    car.speed -= 2 # slow down
            else: 
                car.speed += 2 # speed up
        # check if the car is still alive
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map) # update the car
                genomes[i][1].fitness += car.get_reward() # update the fitness
        if still_alive == 0:
            break # exit loop if no car is alive
        counter += 1
        if counter == 30 * 40: # stop after every 20 seconds
            break
        # draw map and all cars that are alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        # display generation and alive info
        text = generation_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (429, 180)
        screen.blit(text, text_rect)
        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (429, 230)
        screen.blit(text, text_rect)
        pygame.display.flip() # update the display
        clock.tick(60) # 60 frames per second

if __name__ == "__main__":
    # load the NEAT config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path)
    # create a population and add reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    # run the simulation for a maximum of 1000 generations
    population.run(run_simulation, 1000)

        



