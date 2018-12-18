import numpy as np
from simulater import Block as bl
from simulater import Space as sp
from xml.etree.ElementTree import parse
import csv
import datetime
import pandas as pd
import scipy.stats as stats
import re

stock_scale = 1/0.4
arrival_scale = 1/0.22

def import_data(strFilePath, width, height, num_block):
    strFilePath = '../data/E130_BLF결과2.csv'
    if strFilePath == None:
        return generate_space_block(width, height, num_block)
    splitted = strFilePath.split('.')
    if splitted[-1] == 'csv':
        return import_data_from_csv(strFilePath, width, height)
    elif splitted[-1] == 'xml':
        return import_data_from_xml(strFilePath)
    else:
        return None

def import_data_from_xml(strFilePath):
    tree = parse(strFilePath)
    shape = tree.getroot()
    spaces = shape.getchildren()
    space = spaces[0]
    blocks=space.getchildren()
    lblocks=[]
    for block in blocks:
        _height=int(block.get('height'))
        _width = int(block.get('width'))
        _input = int(block.get('input'))
        _output = int(block.get('output'))
        _name = block.get('name')
        lblocks.append(bl.Block(_width, _height, _input, _output, _name))
    space_h=int(space.get('height'))
    space_w=int(space.get('width'))
    space_name=space.get('name')
    cspace=sp.Space(space_h, space_w, space_name)
    return cspace, lblocks

def import_data_from_csv(strFilePath, width, height):
    cspace = sp.Space(width, height, 'test_area')
    lblocks = []
    with open(strFilePath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        s=0
        for row in spamreader:
            if s != 0:
                date_in = datetime.datetime.strptime(row[1], '%Y-%m-%d')
                date_out = datetime.datetime.strptime(row[3], '%Y-%m-%d')
                fix = -1
                if len(row) > 4:
                    fix = max(-1, int(row[4]) - 1)
                lblocks.append(bl.Block(1, 1, date_in, date_out, row[0], fix))
            s +=1
    return cspace, lblocks

def export_2darray_csv(result, strFilePath):
    with open(strFilePath, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for i in range(result.shape[0]):
            spamwriter.writerow(result[i])

def generate_space_block(width, height, num_block):
    space = sp.Space(width, height, 'test_area')
    df_schedule = generate_schedule(stock_scale, arrival_scale, num_block)
    blocks = []
    for index, row in df_schedule.iterrows():
        date_in = row['시작']
        date_out = row['끝']
        blocks.append(bl.Block(1, 1, date_in, date_out, row['운반대상ID']))
    return space, blocks

def generate_blocks(num_block):
    df_schedule = generate_schedule(stock_scale, arrival_scale, num_block)
    blocks = []
    for index, row in df_schedule.iterrows():
        date_in = row['시작']
        date_out = row['끝']
        blocks.append(bl.Block(1, 1, date_in, date_out, row['운반대상ID']))
    return blocks

def generate_schedule(stock_scale, arrival_scale, num_block = 50):
    df_schedule = pd.DataFrame(columns=['운반대상ID', '지번', '시작', '끝', '기간'])
    arrivals = stats.expon.rvs(scale=arrival_scale, size=num_block)
    stocks = stats.expon.rvs(scale=stock_scale, size=num_block)
    current_time = datetime.datetime.strptime('2018-03-01 09:00:00', '%Y-%m-%d %H:%M:%S')
    j = 0
    for i in range(num_block):
        next_time = current_time + datetime.timedelta(hours=arrivals[i])
        end_time = next_time + datetime.timedelta(days=stocks[i])
        duration = datetime.timedelta(days=end_time.day - next_time.day)
        current_time = next_time
        if duration.days == 0:
            duration = datetime.timedelta(days=1)
            end_time += duration
        next_time = datetime.datetime(next_time.year, next_time.month, next_time.day)
        end_time = datetime.datetime(end_time.year, end_time.month, end_time.day)
        j += 1
        row = pd.Series(['block' + str(j), 'yard', next_time, end_time, duration],
                        index=['운반대상ID', '지번', '시작', '끝', '기간'])
        df_schedule = df_schedule.append(row, ignore_index=True)
    return df_schedule


