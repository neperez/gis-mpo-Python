import pygame
import csv
import sys
import os

class Viz_MPO:
    #class constructor:
    #constants, initial values, pygame initialization                           
    def __init__(self):
        #State constants
        self.STATE_WAITING = 0
        self.STATE_PLAY = 1
        self.STATE_MARK = 2
        self.STATE_STOP = 3
    
        #Color constants
        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)
        self.RED = (255,0,0)
        self.GRAY = (211,211,211)
        self.MPO_COLOR = 0

        #Misc. constants
        self.SCREEN_SIZE = 600,600
        self.x_off = -6
        self.y_off = -8
        
        #file handling
        self.file_name = ''
        self.ifile = ''
        self.ofile = ''
        self.reader = ''
        self.writer = ''
        self.row = ''
        self.row_length = 0
        self.file_name = ''
        self.row_index = 0
        self.paths = []
        self.index_mark = []
        self.cont = True

        #pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)
        pygame.display.set_caption("MPO Pattern Visualizer and Marker")
        self.clock = pygame.time.Clock()
        
        # font is changed here
        if pygame.font:
            self.font = pygame.font.Font(None, 30)
        else:
            self.font = None

    #change state based on pressed key
    def check_input(self):
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_SPACE]:
            if self.state == self.STATE_PLAY:
                self.state = self.STATE_MARK
                
        if keys[pygame.K_p]:
            if self.state == self.STATE_WAITING:
                self.state = self.STATE_PLAY
                
        if keys[pygame.K_q]:
            pygame.quit()

    #switches between landmark row(0) and data row(1)
    def switch_row(self,row_type):
        self.ifile.close()
        self.ifile = open(self.file_name)
        self.reader = csv.reader(self.ifile)
        self.row = self.reader.next()

        if row_type == 1:
            self.row = self.reader.next()

    #displays landmarks
    def draw_landmarks(self):
        for i in range(0,len(self.row)-1,3):
            if not self.row[i].isdigit(): #prevents empty values
                break

            #get current landmark info and draw
            x,y,r = int(self.row[i]), int(self.row[i+1]), int(self.row[i+2])
            points = x,y
            lndmrk = pygame.draw.circle(self.screen, self.BLACK, points, r)
            x = points[0]+self.x_off
            y = points[1]+self.y_off
            pts = x,y

            # write landmark number in center of landmark
            font_surface = self.font.render(str(int(i/3)+1),False,self.RED)
            self.screen.blit(font_surface,pts)

    def list2FloatPairs(self,in_list):
        n = len(in_list)
        out_list = []
        
        for i in range(0,n-1,2):
            if not (in_list[i].isalpha() or in_list[i]==''):
                out_list.append((float(in_list[i]),float(in_list[i+1])))

        return out_list
    
    # draws path of the mpo represented by dots
    def draw_path(self):
        path = self.list2FloatPairs(self.row)
        
        for pt in path:
            x = int(pt[0])
            y = int(pt[1])
            pts = x,y
            self.screen.set_at(pts,self.BLACK)

    def show_controls(self):
        print('\nControls: ')
        print('     p - begin playback of MPO pattern')
        print('     Spacebar - Mark index of desired subSequence')
        
    #draws red circle representing MPO
    def draw_mpo(self):
        x = int(self.row[self.row_index])
        y = int(self.row[self.row_index + 1])
        pt = x,y
        color = 0
        
        if self.MPO_COLOR == 0:
            color = self.RED
        else:
            color = self.WHITE
        pygame.draw.circle(self.screen,color,pt,10)
        
    #main loop
    def run(self,fname):
        # draw initial scene
        self.file_name = fname
        self.ifile = open(self.file_name)
        self.reader = csv.reader(self.ifile)
        self.row = self.reader.next()
        self.draw_landmarks()
        self.switch_row(1)
        self.row_length = len(self.row)
        self.draw_path()
        self.draw_mpo()
        
        self.state = self.STATE_WAITING
        self.show_controls()
        
        while (1):
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.ifile.close()
                    pygame.quit()
                    sys.exit()

            self.screen.fill(self.GRAY)

            self.check_input()

            # if waiting, draw initial state
            if self.state == self.STATE_WAITING:
                self.switch_row(0)
                self.draw_landmarks()
                self.switch_row(1)
                self.draw_path()
                self.draw_mpo()
                self.clock.tick(20)

            # draw mpo along path
            if self.state == self.STATE_PLAY:
                self.cont = True
                if self.row_index <= self.row_length - 3:
                    self.switch_row(0)
                    self.draw_landmarks()
                    self.switch_row(1)
                    self.draw_path()
                    if self.row[self.row_index].isdigit():
                        self.draw_mpo()
                        self.row_index += 2
                        self.clock.tick(20)
                    else:
                        self.state = self.STATE_STOP
            
            if self.state == self.STATE_MARK:
                    if self.cont:
                        self.index_mark.append(self.row_index)
                        if self.MPO_COLOR == 0:
                            self.MPO_COLOR = 1
                        else:
                            self.MPO_COLOR = 0
                        self.cont = False
                    self.state = self.STATE_PLAY
                    
            #write indexes to train_file and stop program    
            if self.state == self.STATE_STOP:
                
                #create new training file, and append indices
                subfile_name = self.file_name.split('\\')[-1].split('.')[0]
                new_name = 'train_' + subfile_name + '.csv'
                new_fname = os.path.join(os.path.split(self.file_name)[0],new_name)
                row_to_write = []
                for ind in self.index_mark:
                    row_to_write.append(ind)

                # get type of visit for each landmark
                for i in range(0,len(self.row)-1,1):
                    if self.row[i].isalpha():
                        row_to_write.append(self.row[i])

                # write to new file
                self.ofile = open(new_fname,'wb')
                self.writer = csv.writer(self.ofile, delimiter=',')
                self.writer.writerow(row_to_write)

                #clean and quit
                self.ofile.close()
                self.ifile.close()
                pygame.quit()
                break
                
            pygame.display.flip()
