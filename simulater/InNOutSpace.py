import numpy as np
import random

class Space:
    def __init__(self, width, height, num_block=0, goal=1, block_indices=[], target=-1, allocation_mode=False):
        self.width = width
        self.height = height
        self.grab = -1
        self.stage = 0
        self.mode = goal
        self.goal = goal
        #goal이 0인 경우는 블록 반출, 1인 경우는 반입, 2는 랜덤
        if goal == 2:
            self.goal = random.randint(0, 1)
        self.is_grab = False
        self.action_space = Action(5)
        self.ale = Ale()
        self.stopped = 0
        self.block_moves = 0
        #블록 배치에 연동시에는 아래 값을 True로 사용
        self.allocation_mode = allocation_mode
        blocks = []
        self.num_block=num_block
        #블록의 위치가 정해진 경우와 랜덤인 경우를 나눠서 block의 좌표 값을 입력
        if type(num_block) is not int:
            nblock = random.randint(num_block[0], num_block[1])
            block_indices = random.sample(range((width) * height), nblock)
            for index in block_indices:
                x = index % (width)
                y = (int)(index / (width))
                blocks.append(np.array([x, y]))
        else:
            for sample in block_indices:
                x, y = sample[0], sample[1]
                blocks.append(np.array([x, y]))
        #if len(blocks) == 0:
          #  blocks.append(np.array([-1, -1]))
        self.blocks = blocks
        self.road = []
        for i in range(height):
            self.road.append(np.array([width - 1, i]))
        self.car = self.road[0]
        self.target = target
        self.reward = 0
        if self.goal == 1:
            if allocation_mode:
                #배치 모드에서는 타겟이 정해져 있고 도로 위에 타겟 블록을 추가해서 사용
                self.target_pos = target
                self.blocks.insert(0, self.road[0])
            else:
                #랜덤 모드에서는 0번째 랜덤 블록 위치를 타겟으로 사용, 타겟 블록은 도로에서 시작
                self.target_pos = blocks[0]
                self.blocks[0] = self.road[0]
            self.grab_block()
            self.stage += -1

    def get_state(self):
        empty = -1
        emplty_road = -2
        target = 0
        normal_block = 1
        target_block = 2
        normal_ontarget = 3
        tp = 4
        tp_onblock = 5
        tp_onblock_target = 6
        tp_carrying = 7
        tp_carrying_target = 8


        state = np.full([self.height, self.width], empty)
        i = 0
        #goal=1인 경우는 블록을 반입, 0인 경우는 반출
        for road in self.road:
            state[road[1], road[0]] = target
        if self.goal == 1:
            for road in self.road:
                state[road[1], road[0]] = emplty_road
            state[self.target_pos[1], self.target_pos[0]] = target
        for block in self.blocks:
            if i == self.target:
                state[block[1], block[0]] = target_block
            else:
                if state[block[1], block[0]] == target:
                    state[block[1], block[0]] = normal_ontarget
                else:
                    state[block[1], block[0]] = normal_block
            i += 1
        #블록이 없는 경우
        if state[self.car[1], self.car[0]] < normal_block:
            state[self.car[1], self.car[0]] = tp
        else:
            if state[self.car[1], self.car[0]] == normal_block:
                if self.is_grab:
                    state[self.car[1], self.car[0]] = tp_carrying
                else:
                    state[self.car[1], self.car[0]] = tp_onblock
            elif state[self.car[1], self.car[0]] == target_block:
                if self.is_grab:
                    state[self.car[1], self.car[0]] = tp_carrying_target
                else:
                    state[self.car[1], self.car[0]] = tp_onblock_target
        return state

    def is_movable(self, x_change, y_change):
        movable = True
        x = self.car[0] + x_change
        y = self.car[1] + y_change
        if x_change==0 and y_change==0:
            movable = False
        elif abs(x_change) + abs(y_change) > 1:
            movable = False
        elif x > self.width - 1:
            movable =False
        elif x < 0:
            movable = False
        elif y > self.height - 1:
            movable = False
        elif y < 0:
            movable = False
        if self.grab > -1:
            for block in self.blocks:
                if block[0] == x and block[1] ==y:
                    movable = False
                    break
        return movable

    def move_car(self, x_change, y_change):
        x = self.car[0] + x_change
        y = self.car[1] + y_change
        self.car = np.array([x, y])
        if self.grab > -1:
            self.blocks[self.grab] = np.array([x, y])
        self.stage += 1
        if self.goal == 0:
            for _road in self.road:
                if np.array_equal(self.blocks[self.target], _road):
                    #self.reward = 1
                    return True
        elif self.goal == 1:
            if np.array_equal(self.blocks[self.target], self.target_pos):
                on_road = False
                for i in range(len(self.blocks)):
                    if i == self.target:
                        continue
                    for road in self.road:
                        if np.array_equal(self.blocks[i], road):
                            on_road = True
                if on_road:
                    return False
                #self.reward = 1
                return True
        return False

    def grab_block(self):
        if self.grab != -1:
            return
        i = 0
        for block in self.blocks:
            if np.array_equal(block, self.car):
                self.grab = i
                self.stage += 1
                self.is_grab = True
                self.block_moves += 1
                #if self.grab == 0:
                #    self.reward = 0.5
                return
            i += 1


    def release_block(self):
        if self.grab != -1:
            self.grab = -1
            self.stage += 1
            self.is_grab = False

    def step(self, action):
        #print(self.get_state())
        max_reward = 30
        self.reward = 0
        x_change = 0
        y_change = 0
        if action == 0:
            y_change += -1
        elif action == 1:
            y_change += 1
        elif action == 2:
            x_change += -1
        elif action == 3:
            x_change += 1
        elif action == 4:
            if self.is_grab:
                self.release_block()
            else:
                self.grab_block()
        terminal = False
        if self.is_movable(x_change, y_change):
            if self.move_car(x_change, y_change):
                #self.reward = max_reward - self.stage
                #if self.reward < 10:
                #    self.reward = 10
                self.reward = 1
                terminal = True
            self.stopped = 0
        else:
            self.stage += 1
            self.stopped += 1
        #self.reward += -0.1
        state = self.get_state()
        reward = self.reward

        if self.stopped > 5 and not self.allocation_mode:
            terminal = True
            #reward = -1
        #if self.reward == 1:
            #terminal = True
        if terminal:
            #print(self.block_moves)
            #print(self.stage)
            if not self.allocation_mode:
                self.__init__(self.width, self.height, self.num_block, self.mode, target=self.target)
        '''
        print(action)
        print([x_change, y_change])
        print(self.is_grab)
        print(state)
        print(reward)
        print(terminal)
        print('*'*30)
        '''
        return state, reward, terminal

class Action:
    def __init__(self, n):
        self.n = n

class Ale:
    def lives(self):
        return 1
