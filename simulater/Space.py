#import objects.Block
import numpy as np
from copy import deepcopy
import datetime

class Space:

    def __init__(self, width, height, name):
        self.width=width
        self.height=height
        self.name=name
        self.RESULTS=[]
        self.status = np.full((self.width, self.height), -1.0, dtype=float)

    #공간에 할당된 블록 정보, 투입 및 반출 일정 정보를 업데이트하는 함수
    def update_blocks(self, blocks):
        events = []
        event_in=[]
        event_out=[]
        for block in blocks:
            start, end = block.get_schedule()
            events.append(start)
            events.append(end)
            event_in.append(start)
            event_out.append([end, block])
        if isinstance(events[0], datetime.date):
            self.TIMETYPE = datetime.date
        else :
            self.TIMETYPE = int
        events = list(set(events))
        events.sort()
        self.EVENTS = events
        self._blocks=blocks
        self.event_in=event_in
        self.event_out=event_out

    #좌표 기준으로 블록을 배치하고 reward를 계산하는 함수
    def update_status(self, stage):
        if stage!=0: #투입 일정 사이에 반출되는 블록을 상태에 반영
            self._transfer_blocks(stage)
        block=self._blocks[stage]
        r=0
        block.isin = True
        width, height = block.get_dimension()
        xloc, yloc = block.get_location()
        bounds = self._make_boundary([xloc, yloc], [width, height])
        for bound in bounds:
            if self.status[bound[0], bound[1]] == 1:
                r += 1
        for i in range(width):
            for j in range(height):
                if (xloc + i < self.status.shape[0] and yloc + j < self.status.shape[1]):
                    self.status[xloc + i, yloc + j] += 1.0
                else:
                    if (i == 0):
                        yloc += -1
                        self.status[xloc + i, yloc + j] += 1.0
                    else:
                        xloc += -1
                        self.status[xloc + i, yloc + j] += 1.0
        self.RESULTS.append(deepcopy(self.status))
        arrangible = True
        for a in np.nditer(self.status):
            if (a > 1):
                arrangible = False
                r=0
                break
        return arrangible, r

    #지번 단위로 블록을 배치하고 reward를 계산하는 함수
    def update_state_lot(self, stage):
        #state = deepcopy(self.get_status(max(0, stage - 1)))
        state = np.full((self.width, self.height), -1.0, dtype=float)
        current_day = self.event_in[stage]
        if self.TIMETYPE is datetime.date:
            isdate = True
        if stage!=0: #투입 일정 사이에 반출되는 블록을 상태에 반영, 배치된 블록의 남은 일수를 업데이트
            #self._transfer_blocks(stage)
            '''
            term = self.event_in[stage]-self.event_in[stage-1]
            if isdate:
                term = float(term.days)
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    if state[i, j] == 0.0:
                        state[i, j] = -1.0
                    elif state[i, j] != -1.0:
                        state[i, j] -= term
                        if state[i, j] <= 0.0:
                            state[i, j] = -1.0
            '''
            blocks = self.get_blocks()
            for i in range(stage):
                _block = blocks[i]
                _start, _end = _block.get_schedule()
                if _start <= current_day < _end:
                    _x, _y = _block.get_location()
                    state[_x, _y] = (_end - current_day).days
        block = self._blocks[stage]
        start, end = block.get_schedule()
        untilout = block.term
        r=0
        block.isin = True
        width, height = block.get_dimension()
        xloc, yloc = block.get_location()

        if state[xloc, yloc] != -1.0:
            arrangible=False
            state[xloc, yloc] = -2.0
        else:
            state[xloc, yloc] += untilout+1
            arrangible = True
        '''
        exitside, otherside = self.separate_area(self.status, [xloc, yloc], 1)
        for lot in exitside:
            if lot<untilout:
                r+=1
            elif lot>untilout:
                r+=-1
        for lot in otherside:
            if lot<untilout:
                r+=-1
            elif lot>untilout:
                r+=1
        '''
        # reward는 1,0,-1 중 하나가 되도록 scaling
        if r > 0:
            #r = int(r/4)+1
            r=1
        elif r < 0:
            #r = int(r/4)-1
            r=-1
        #Result에서 reward 제거
        #self.RESULTS.append([deepcopy(self.status), r])
        #temp = np.zeros(self.status.shape)
        #np.copyto(temp, self.status)
        self.RESULTS.append(state)
        return arrangible, r

    #space를 location을 기준으로 exit(0,1,2,3) 방향 구역과 반대 방향 구역으로 나눔
    def separate_area(self, state, location=[0,0], exit=0):
        #출입구의 방향에 따라서 state와 location을 회전
        if exit==1:
            location = [location[1], state.shape[0]-location[0]-1]
        elif exit==2:
            location = [state.shape[0]-location[0]-1, state.shape[1]-location[1]-1]
        elif exit==3:
            location = [state.shape[1] - location[1]-1, location[0]]
        state = np.rot90(state, -exit)
        exitside=[]
        otherside=[]
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i,j] != -1 and j != location[1]:
                    if j<location[1]:
                        exitside.append(state[i,j])
                    else:
                        otherside.append(state[i,j])
        return exitside, otherside

    #이전 stage 이후 현 stage 이전의 반출 이벤트를 처리하는 함수
    def _transfer_blocks(self, stage):
        for day, block in self.event_out:
            if day>self.event_in[stage-1] and day<=self.event_in[stage]:
                width, height = block.get_dimension()
                xloc, yloc = block.get_location()
                for i in range(width):
                    for j in range(height):
                        if (xloc + i < self.status.shape[0] and yloc + j < self.status.shape[1]):
                            self.status[xloc + i, yloc + j] = 0.0
                        else:
                            if (i == 0):
                                yloc += -1
                                self.status[xloc + i, yloc + j] = 0.0
                            else:
                                xloc += -1
                                self.status[xloc + i, yloc + j] = 0.0
                #self.RESULTS.append([deepcopy(self.status), 0])

    def set_status(self, date):
        self.CURRENTDATE = date
        status = np.full((self.status.shape[0], self.status.shape[1]), .0, dtype=float)
        #status=self.status
        for block in self._blocks:
            startdate, enddate = block.get_schedule()
            if (date >= startdate and date < enddate):
                width, height = block.get_dimension()
                xloc, yloc = block.get_location()
                for i in range(width):
                    for j in range(height):
                        if (yloc + j < status.shape[0] and xloc + i < status.shape[1]):
                            status[yloc + j, xloc + i] += 1.0
                        else:
                            if (j == 0):
                                xloc += -1
                                status[yloc + j, xloc + i] += 1.0
                            else:
                                yloc += -1
                                status[yloc + j, xloc + i] += 1.0
        self.RESULTS.append(status)
        arrangible = True
        for a in np.nditer(status):
            if (a > 1):
                arrangible = False
                break
        return arrangible

    def get_status(self, stage):
        '''
        day=self.event_in[stage]
        if (day >= self.EVENTS[-1]):
            return self.RESULTS[-1]
        for i in range(len(self.EVENTS)-1):
            if(self.EVENTS[i]<=day and self.EVENTS[i+1]>day):
                index=i
                break
        if(len(self.RESULTS)==0):
            status=np.full((self.width, self.height), -1.0, dtype=float)
        else:
            if(len(self.RESULTS)<=index):
                print('here')
            status=self.RESULTS[index]
        '''
        if (len(self.RESULTS) == 0):
            status = np.full((self.width, self.height), -1.0, dtype=float)
        else:
            status = self.RESULTS[stage]
        return status

    def get_blocks(self):
        return self._blocks

    def _make_boundary(self, location, size):
        bounds=[]
        if location[1] != 0:
            for i in range(size[0]):
                if location[0] + i< self.width:
                    bounds.append([location[0]+i, location[1]-1])
        if location[0] + size[0] < self.width:
            for i in range(size[1]):
                if location[1] + i <self.height:
                    bounds.append([location[0] + size[0], location[1]+i])
        if location[1] + size[1] < self.height:
            for i in range(size[0]):
                if location[0] + i < self.width:
                    bounds.append([location[0]+i, location[1] + size[1]])
        if location[0] != 0:
            for i in range(size[1]):
                if location[1] + i < self.height:
                    bounds.append([location[0]-1, location[1]+i])
        return bounds

    def modify_latest_state(self, new_state):
        if len(self.RESULTS) > 0:
            self.RESULTS[-1] = new_state







