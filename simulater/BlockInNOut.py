import pygame
import numpy as np
import random
import time
from simulater.InNOutSpace import Space

class BlockInNOut:
    white = (255,255,255)
    black = (0,0,0)

    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    dark_red = (200, 0, 0)
    dark_green = (0, 200, 0)
    dark_blue = (0, 0, 200)

    x_init = 100
    y_init = 100
    x_span = 100
    y_span = 100
    thickness = 5

    display_width = 1000
    display_height = 600
    gameDisplay = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('Block In & Out')
    clock = pygame.time.Clock()
    pygame.key.set_repeat()
    button_goal = (display_width - 100, 10, 70, 40)

    def __init__(self, width, height, num_block, goal):
        self.grab = False
        self.width = width
        self.height = height
        self.num_block = num_block
        self.goal = goal
        self.space = Space(width, height, num_block, goal)
        self.on_button = False

    def restart(self):
        self.grab = False
        if self.space.reward > 0:
            self.message_display(str(self.space.step) + 'steps')
        self.space = Space(width, height, self.num_block, self.goal)
        self.game_loop_from_space()

    def text_objects(self, text, font):
        textSurface = font.render(text, True, self.white)
        return textSurface, textSurface.get_rect()

    def car(self, x, y):
        pygame.draw.circle(self.gameDisplay, self.red, ((int)(self.x_init + (x + 0.5) * self.x_span),
                                              (int)(self.y_init + (y + 0.5) * self.y_span)), (int)(self.x_span/2 - 10))
        pygame.draw.circle(self.gameDisplay, self.black, ((int)(self.x_init + (x + 0.5) * self.x_span),
                                                        (int)(self.y_init + (y + 0.5) * self.y_span)),
                           (int)(self.x_span / 2 - 9), 2)
        largeText = pygame.font.Font('freesansbold.ttf', 20)
        TextSurf, TextRect = self.text_objects('T/P', largeText)
        TextRect.center = ((int)(self.x_init + self.x_span * (x + 0.5)),
                           (int)(self.y_init + self.y_span * (y + 0.5)))
        self.gameDisplay.blit(TextSurf, TextRect)

    def block(self, x, y, text='', color=(0,255,0)):
        if text == 'block0':
            color = self.red
        pygame.draw.rect(self.gameDisplay, color, ((int)(self.x_init + self.x_span * x),
                                                        (int)(self.y_init + self.y_span * y),
                                                        (int)(self.x_span),
                                                        (int)(self.x_span)))
        largeText = pygame.font.Font('freesansbold.ttf', 20)
        TextSurf, TextRect = self.text_objects(text, largeText)
        TextRect.center = ((int)(self.x_init + self.x_span * (x + 0.5)),
                           (int)(self.y_init + self.y_span * (y + 0.5)))
        self.gameDisplay.blit(TextSurf, TextRect)

    def board(self, step, reward=0, grab=-1):
        largeText = pygame.font.Font('freesansbold.ttf', 20)

        TextSurf, TextRect = self.text_objects('step: ' + str(step) +
                                               '   reward: ' + str(reward) +
                                               '   carrying: ' + str(grab), largeText)
        TextRect.center = (200, 20)
        self.gameDisplay.blit(TextSurf, TextRect)

    def button(self, goal = 0):
        color = self.dark_blue
        str_goal = 'In'
        if self.on_button:
            color = self.blue
        if goal == 0:
            str_goal = 'Out'
            color = self.dark_red
            if self.on_button:
                color = self.red
        pygame.draw.rect(self.gameDisplay, color, self.button_goal)
        largeText = pygame.font.Font('freesansbold.ttf', 20)
        TextSurf, TextRect = self.text_objects(str_goal, largeText)
        TextRect.center = (int(self.button_goal[0] + 0.5 * self.button_goal[2]),
                           int(self.button_goal[1] + 0.5 * self.button_goal[3]))
        self.gameDisplay.blit(TextSurf, TextRect)

    def game_loop(self, x, y):

        x_change = 0
        y_change = 0

        gameExit = False

        while not gameExit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        x_change = -1
                    elif event.key == pygame.K_RIGHT:
                        x_change = 1
                    elif event.key == pygame.K_UP:
                        y_change = -1
                    elif event.key == pygame.K_DOWN:
                        y_change = 1
                    break
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        x_change = 0
                    elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        y_change = 0

            x += x_change
            y += y_change

            if x > self.width - 1:
                self.game_loop(x - 1, y)
            elif x < 0:
                self.game_loop(x + 1, y)
            elif y > self.height - 1:
                self.game_loop(x, y - 1)
            elif y < 0:
                self.game_loop(x, y + 1)

            self.gameDisplay.fill(self.black)
            self.draw_grid()
            self.block(0, 0)
            self.car(x,y)
            pygame.display.flip()
            self.clock.tick(10)

    def game_loop_from_space(self):
        x_change = 0
        y_change = 0

        gameExit = False
        state = np.zeros([self.width, self.height])
        while not gameExit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        x_change = -1
                    elif event.key == pygame.K_RIGHT:
                        x_change = 1
                    elif event.key == pygame.K_UP:
                        y_change = -1
                    elif event.key == pygame.K_DOWN:
                        y_change = 1
                    elif event.key == pygame.K_SPACE:
                        if self.grab:
                            self.grab = False
                            self.space.release_block()
                        else:
                            self.grab = True
                            self.space.grab_block()
                    elif event.key == pygame.K_ESCAPE:
                        gameExit = True
                        break
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        x_change = 0
                    elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        y_change = 0

                click = pygame.mouse.get_pressed()
                mouse = pygame.mouse.get_pos()
                self.on_button = False
                if self.button_goal[0] < mouse[0] < self.button_goal[0] + self.button_goal[2]:
                    if self.button_goal[1] < mouse[1] < self.button_goal[1] + self.button_goal[3]:
                        self.on_button = True
                        if click[0] == 1:
                            if self.goal == 0:
                                self.goal = 1
                            else:
                                self.goal = 0
                            self.restart()

            if self.space.is_movable(x_change, y_change):
                self.space.move_car(x_change, y_change)

            self.gameDisplay.fill(self.black)
            self.draw_road(self.space.road)
            self.draw_space(self.space)
            self.draw_grid()
            if np.array_equal(state, self.space.get_state()) == False:
                print(self.space.get_state())
            state = self.space.get_state()
            pygame.display.flip()
            self.clock.tick(10)

            if self.space.reward > 0:
                self.restart()

    def draw_road(self, road):
        i = 0
        for _road in road:
            self.block(_road[0], _road[1], 'road', self.blue)
            i += 1
    def draw_grid(self):
        width = self.width
        height = self.height
        pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init),
                         (self.x_init, self.y_init + self.y_span * height), self.thickness)
        pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init),
                         (self.x_init + self.x_span * width, self.y_init), self.thickness)

        for i in range(width):
            pygame.draw.line(self.gameDisplay, self.blue, (self.x_init + self.x_span * (i + 1), self.y_init),
                             (self.x_init + self.x_span * (i + 1), self.y_init + self.y_span * height), self.thickness)
        for i in range(height):
            pygame.draw.line(self.gameDisplay, self.blue, (self.x_init, self.y_init + self.y_span * (i + 1)),
                             (self.x_init + self.x_span * width, self.y_init + self.y_span * (i + 1)), self.thickness)
    def draw_space(self, space):
        i = 0
        for block in space.blocks:
            self.block(block[0], block[1], 'block'+str(i))
            i += 1
        self.car(space.car[0], space.car[1])
        self.board(space.stage, space.reward, space.grab)
        self.button(space.goal)
        if space.goal == 1:
            self.block(space.target_pos[0], space.target_pos[1], 'target', self.blue)

    def message_display(self, text):
        largeText = pygame.font.Font('freesansbold.ttf', 115)
        TextSurf, TextRect = self.text_objects(text, largeText)
        TextRect.center = (int(self.display_width / 2), int(self.display_height / 2))
        self.gameDisplay.blit(TextSurf, TextRect)
        pygame.display.update()
        time.sleep(1)

random.seed(2)
width = 7
height = 4
num_block = 15
pygame.init()
a = BlockInNOut(width, height, num_block, 0)
#space = Space(width, height, 12)
#a.game_loop(0, 0)
a.game_loop_from_space()
quit()

