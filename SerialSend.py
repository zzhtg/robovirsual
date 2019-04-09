import cv2
import serial
import serial.tools.list_ports
import binascii

MODE = {'PITCH': 0x11, 'YAW': 0x12} # choose which Axis of Stm32 Robocloud to send
def Serial_Send(ser, raw_pitch, raw_yaw):
    Msg = Process(raw_pitch, raw_yaw)
    ser.write(Msg.encode('ascii'))
    return Msg

def Serial_init(boudrate, Timeout):
    # Get A initial Serial, or return a zero-value and exit python
    port_list = list(serial.tools.list_ports.comports()) 
    if len(port_list) <= 0: 
        print("The Serial port can't find!")
        return 0
    else:
        port_list_0 = list(port_list[0])
        port_serial = port_list_0[0] 
        ser = serial.Serial(port_serial, boudrate, timeout = Timeout)
        print("check which port was really used >",ser.name)
        return ser

def Process(raw_pitch, raw_yaw):
    # Processing the message to STM32, including UTF-8 to ascii and frame head, tail and checksum 
    re_frame = chr(MODE['PITCH'])
    pitch = str(raw_pitch)
    pitch = '0' * (3 - len(pitch)) + pitch
    re_frame += pitch
    re_frame += chr(MODE['YAW'])
    yaw = str(raw_yaw)
    yaw = '0' * (3 - len(yaw)) + yaw
    re_frame += yaw
    checksum = MODE['PITCH'] + MODE['YAW']
    len_str = len(re_frame)
    for i in range(0, len_str):
        if not(i == 0 or i == 4):
            checksum += int(re_frame[i]) + 0x30
    checksum %= 255
    re_frame += chr(checksum)
    return re_frame

if(__name__ == "__main__"):
    ser = Serial_init(115200, 1)
    if(ser == 0):
        print("Caution: Serial Not Found!") # print caution
    else:
        while(1):
            raw_pitch = input("Please input the pitch number: \n")
            raw_yaw = input("Please input the yaw number: \n")
            if(raw_pitch == 'q' or raw_yaw == 'q'):
                break;
            Msg = Process(raw_pitch, raw_yaw)
            print("You have sent", ord(Msg[0]), Msg[1:3], ord(Msg[4]), Msg[5:-2], ord(Msg[-1]), 'to the serial\n')
            ser.write(Msg.encode('ascii'))
            rec = ser.readline()
            print(rec)
        ser.close()
    
