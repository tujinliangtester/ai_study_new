'''
Python | 读取 midi 文件
https://www.jianshu.com/p/931be4f387bb

'''

from struct import unpack
import time


def read_vlq(f):
    result = ''
    buffer = unpack('B', f.read(1))[0]
    length = 1
    while buffer > 127:
        print(buffer)
        result += '{0:{fill}{n}b}'.format(buffer - 128, fill='0', n=7)
        buffer = unpack('B', f.read(1))[0]
        length += 1

    result += '{0:{fill}{n}b}'.format(buffer, fill='0', n=7)
    return int(result, 2), length


def parse_event(evt, param):
    if 128 <= evt <= 143:
        print('Note Off event.')
    elif 144 <= evt <= 159:
        print('Note On event.', unpack('>BB', param))
    elif 176 <= evt <= 191:
        print('Control Change.')
    elif 192 <= evt <= 207:
        print('Program Change.')

fpath='jsbach/BWV532.mid'
with open(fpath, 'rb') as f:
    # print(f.read(200))
    # print(f.read(4))
    # HEADER
    if f.read(4) != b'MThd':
        raise Exception('not a midi file!')
    print(f.read(4))
    header_info = f.read(6)
    print(unpack('>hhh', header_info))

    ''' ================================== '''
    while True:
        track_head = f.read(4)
        if track_head != b'MTrk':
            if track_head != b'':
                print(f.read(20))
                raise Exception('not a midi file!')
            else:
                break

        # length of track
        len_of_track = unpack('>L', f.read(4))[0]
        # print('len_of_track ', len_of_track)

        counter = 0
        t = 0
        last_event = None
        while True:
            delta_t, len_ = read_vlq(f)
            counter += len_
            t += delta_t
            # print('T ', t, end='')
            event_code = f.read(1)
            event_type = unpack('>B', event_code)[0]
            counter += 1
            # print(' event_type ', event_type, end='')
            if event_type == 255:
                meta_type = f.read(1)
                counter += 1
                # print(' - meta_type ', meta_type, end='')
                data_len, len_ = read_vlq(f)
                counter += len_
                data = f.read(data_len)
                counter += data_len
                # print(' - ', data)
            elif event_type <= 127:
                parse_event(last_event, event_code + f.read(1))
                counter += 1
            else:
                if 128 <= event_type <= 143:
                    # print(' Note Off event.', end='')
                    parse_event(event_type, f.read(2))
                    counter += 2
                elif 144 <= event_type <= 159:
                    # print(' Note On event.', end='')
                    parse_event(event_type, f.read(2))
                    counter += 2
                elif 176 <= event_type <= 191:
                    # print(' Control Change.', end='')
                    parse_event(event_type, f.read(2))
                    counter += 2
                elif 192 <= event_type <= 207:
                    # print(' Program Change.', end='')
                    parse_event(event_type, f.read(1))
                    counter += 1
                last_event = event_type

            # print(counter)
            if counter == len_of_track:
                break
