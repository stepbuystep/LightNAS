# -*- coding: utf-8 -*- 

def read_power_xavier(print_detail=False):
    """read power use tralley for Nvidia Jetson Xavier"""
    with open("/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input") as f:
        lines = f.readlines()
        device0_0 = int(lines[0].strip("\n"))
    with open("/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input") as f:
        lines = f.readlines()
        device0_1 = int(lines[0].strip("\n"))
    with open("/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power2_input") as f:
        lines = f.readlines()
        device0_2 = int(lines[0].strip("\n"))
    with open("/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power0_input") as f:
        lines = f.readlines()
        device1_0 = int(lines[0].strip("\n"))
    with open("/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power1_input") as f:
        lines = f.readlines()
        device1_1 = int(lines[0].strip("\n"))
    with open("/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power2_input") as f:
        lines = f.readlines()
        device1_2 = int(lines[0].strip("\n"))
    total_power = device0_0 + device0_1 + device0_2 + device1_0 + device1_1 + device1_2
    if print_detail:
        print("{} = {} + {} + {} + {} + {} + {}".format(total_power, device0_0, device0_1, device0_2, device1_0, device1_1, device1_2))
    
    return total_power
