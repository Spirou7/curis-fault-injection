import struct

def record(train_recorder, text):
    if train_recorder:
        train_recorder.write(text)
        train_recorder.flush()

def bin2fp32(bin_str):
    assert len(bin_str) == 32
    return struct.unpack('!f',struct.pack('!I', int(bin_str, 2)))[0]

def fp322bin(value):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))
