import gtk
import pygame
import csv
import sys
import os

class Viz_MPO:
    """Animates a MPO pattern from a file."""

    def __init__(self):
        """ Viz_MPO class constructor

        Initializes all globals, constants and states
        needed for program function.

        """
        #State constants
        self.STATE_WAITING = 0# default state
        self.STATE_PLAY = 1# lasts until end of row, then reverts to waiting
        self.STATE_MARK = 2# marks sub-trajectory
        self.STATE_STOP = 3# program cleanup before exit
        self.STATE_NEXT = 4# reverts to waiting after next sample row is loaded.
        self.STATE_PREV = 5# reverts to waiting after previous sample row is loaded.
        self.state = self.STATE_WAITING # initial program state

        #Color constants
        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)
        self.RED = (255,0,0)
        self.GRAY = (211,211,211)
        self.GREEN = (0,255,0)

        # MPO info
        self.mpo_color = 0 # red (0) if in normal trajectory section, white(1) if in sub-trajectory.
        self.first_index = True # indicates if the first index of sub-trajectory has been chosen.
        self.first = 0 # index of first sub-trajectory point
        self.second = 0 # index of second sub-trajectory point
        self.sub_index_list = [] # format is: [(number of data row, start index in row, end index in row),...]

        # Landmark info
        self.num_lndmrks = 0# number of landmarks in the file
        self.bg_img_path = '' # background image path
        self.PATTERN_INFO_ROW = 0 # index of information about pattern
        self.LANDMARK_VERTEX_ROW = 1 # index of first row containing landmark vertices
        self.BG_IMAGE_INDEX = 0 # index in row where background image path is stored.
        self.LNDMRK_NUM_INDEX = 3 # index in row where number of landmarks is specified

        # File handling
        self.file_name = '' # path of file name
        self.ifile = '' # file to read
        self.ofile = '' #file to write
        self.reader = '' # csv reader object
        self.writer = '' # csv writer object
        self.bg = False # flag if file has a bg image.

        # Row info
        self.row = '' # current row from file
        self.row_length = 0 # length of current row 
        self.tot_rows = 0 # total amount of rows in file
        self.data_start_row = 0 # row number where path data starts
        self.data_row_num = 0 # number of current data row
        self.paths = [] # holds mpo path data
        self.row_index = 0 # holds current mpo position
        self.tot_data_rows = 0

        # Pygame info
        self.screen_width = 600 # default values
        self.screen_height = 600 # default values
        self.SCREEN_SIZE = self.screen_width, self.screen_height                           
        self.WIDTH_INDEX = 1 # index in pattern info row that specifies window width
        self.HEIGHT_INDEX = 2 # index in pattern info row that specifies window height

    def get_file_name(self):
        """ Uses gtk to gather file name from user. """
        dialog = gtk.FileChooserDialog("Open..",
                                       None,
                                       gtk.FILE_CHOOSER_ACTION_OPEN,
                                       (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                        gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)
        response = dialog.run()

        if response == gtk.RESPONSE_OK:
            self.file_name = dialog.get_filename()
            self.tot_rows = len(open(self.file_name).readlines())
            self.ifile = open(self.file_name)
            self.reader = csv.reader(self.ifile)
            self.row = self.reader.next()
            dialog.destroy()
        elif response == gtk.RESPONSE_CANCEL:
            print 'Closed, no file selected.'
            dialog.destroy()
            
    def get_file_data(self):
        """ Gathers pertinent data about current file"""
        self.switch_row(self.PATTERN_INFO_ROW)

       # get background image file path
        if self.row[0] != 'No image':
            # set path to background image, and get/set screen size
            self.bg_img_path = self.row[self.BG_IMAGE_INDEX]
            self.screen_width = int(self.row[self.WIDTH_INDEX])
            self.screen_height = int(self.row[self.HEIGHT_INDEX])
            self.SCREEN_SIZE = self.screen_width, self.screen_height
            self.bg = True
           
        # get number of landmarks
        self.num_lndmrks = int(self.row[self.LNDMRK_NUM_INDEX])

        # get starting row number for mpo path data
        self.data_start_row = self.num_lndmrks + 1
        self.tot_data_rows = self.tot_rows - self.data_start_row
        self.data_row_num = self.data_start_row
                
    def init_pygame(self):
        """Initializes pygame features."""
        pygame.init()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)
        pygame.display.set_caption("MPO Pattern Visualizer and Marker")
        self.clock = pygame.time.Clock()

        # font is changed here
        if pygame.font:
            self.font = pygame.font.Font(None, 30)
        else:
            self.font = None

    def check_input(self):
        """Changes program state based on pressed key.

        program state is set here, implementation handled in main loop

        """
        keys = pygame.key.get_pressed()

        # 'SPACE' key demarcates the beginning and end of a MPO sub-trajectory
        if keys[pygame.K_SPACE]:
            if self.state == self.STATE_PLAY:
                self.state = self.STATE_MARK

        # 'ENTER' key animates the currently loaded mpo pattern sample
        if keys[pygame.K_RETURN]:
            if self.state == self.STATE_WAITING:
                self.state = self.STATE_PLAY

        # 'p' key moves to previous mpo pattern descibed in file.
        if keys[pygame.K_p]:
            if self.state == self.STATE_WAITING:
                self.state = self.STATE_PREV
            elif self.state == self.STATE_PLAY:
                self.state = self.STATE_PREV

        # 'n' key moves to next mpo pattern described in file.
        if keys[pygame.K_n]:
            if self.state == self.STATE_WAITING or \
               self.state == self.STATE_PLAY:
                self.state = self.STATE_NEXT

        # 'q' key force quits the program
        if keys[pygame.K_q]:
            self.state = self.STATE_STOP

    def switch_row(self, row_num):
        """Brute force row switch.
        
        Keyword argument:
        row_num -- The row in file to load in the csv reader variable

        """
        self.ifile.close()
        self.ifile = open(self.file_name)
        self.reader = csv.reader(self.ifile)
        self.reader.next()

        if row_num > 0:
            for i in range(0,row_num):
                self.row = self.reader.next()
                
        self.data_row_length = 0
        for element in self.row:
            if element != '':
                self.data_row_length += 1
        
        
    def show_frame_info(self):
        """Displays current data row number on screen"""
        if self.font:
            font_surface = self.font.render("Sample Number: " + \
                                            str(self.data_row_num-self.num_lndmrks) + \
                                            " of " + \
                                            str(self.tot_rows - self.data_start_row), \
                                            False, self.GREEN)
            pts = (self.screen_width / 3),3
            self.screen.blit(font_surface,pts)
            
    def list2FloatPairs(self,in_list):
        """Converts list of integers to list of float pairs.

        Keyword argument:
        in_list -- The list of numbers to convert to float pairs

        """
        n = len(in_list)
        out_list = []
        for i in range(0,n-1,2):
            if not (in_list[i].isalpha() or in_list[i]==''):
                out_list.append((float(in_list[i]),float(in_list[i+1])))
        return out_list
   
    def draw_landmarks(self):
        """Draws all landmarks for current frame."""
        xMin = 0
        xMax = 0
        yMin = 0
        yMax = 0
        xCenter = 0
        yCenter = 0

        for i in range(self.num_lndmrks):
            self.switch_row(i+1)
            lndmrk_vertex = []

            for j in range(0,len(self.row)-3,2):

                if self.row[j] == '' or self.row[j+1] == '':
                    break
                pair = int(self.row[j]), int(self.row[j+1])

                if j==0:
                    xMin = pair[0]
                    xMax = pair[0]
                    yMin = pair[1]
                    yMax = pair[1]
                lndmrk_vertex.append(pair)
                
                # calculate bounding box around landmark
                if int(self.row[j]) < xMin:
                    xMin = int(pair[0])
                if int(self.row[j]) > xMax:
                    xMax = int(pair[0])
                if int(self.row[j+1]) < yMin:
                    yMin = int(pair[1])
                if int(self.row[j+1]) > yMax:
                    yMax = int(pair[1])

                # get center of bounding box.
                xCenter = xMin + ((xMax - xMin) / 2)
                yCenter = yMin + ((yMax - yMin) / 2)
            pygame.draw.polygon(self.screen,self.BLACK,lndmrk_vertex,2)
            pts = xCenter,yCenter

            # write landmark number near 'center' of landmark
            font_surface = self.font.render(str(i+1),False,self.RED)
            self.screen.blit(font_surface,pts)

    def draw_path(self):
        """Draws paths for current frame."""

        # using current data row number
        # switch to appropriate row in file
        self.switch_row(self.data_row_num)
        path = self.list2FloatPairs(self.row)

        for pt in path:
            x = int(pt[0])
            y = int(pt[1])
            pts = x,y
            pygame.draw.circle(self.screen,self.BLACK,pts,2)

    def draw_mpo(self):
        """Draws mpos for current frame."""

        x = int(self.row[self.row_index])
        y = int(self.row[self.row_index + 1])
        pt = x,y
        color = 0
        if self.mpo_color == 0:
            color = self.RED
        else:
            color = self.WHITE # white during desired sub-trajectory
        pygame.draw.circle(self.screen,color,pt,10)
   
    def show_controls(self):
        """Shows controls to user in console."""

        print('\nControls: ')
        print('     ENTER - begin playback of MPO pattern')
        print('     p - go to previous sample')
        print('     n - go to next sample')
        print('     SPACE - Mark index of desired sub-trajectory')
        print('     q - quit program and write results to training file')

    def run(self):
        """Main animation loop."""

        # first run actions
        self.state = self.STATE_WAITING
        self.show_controls()
        self.get_file_name()
        self.get_file_data()
        self.init_pygame()

        while (1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.ifile.close()
                    pygame.quit()
                    sys.exit()

            # clear screen and check for input from keyboard
            if self.bg:
                background = pygame.image.load(self.bg_img_path)
                bgRect = background.get_rect()
                self.screen.blit(background,bgRect)
            else:
                self.screen.fill(self.GRAY)
            self.check_input()

            # handle actions for all states:
            if self.state == self.STATE_WAITING:
                self.draw_landmarks()
                self.draw_path()
                self.draw_mpo()
                self.clock.tick(80)

            # draws current mpo position, and increments index for next frame
            if self.state == self.STATE_PLAY:
                if self.row_index < self.data_row_length - 2:
                    self.row_index += 2
                    self.draw_landmarks()
                    self.draw_path()
                    self.draw_mpo()
                    self.clock.tick(30)
                else:
                    self.state = self.STATE_WAITING

            # loads in next sample row from file
            if self.state == self.STATE_NEXT:
                if self.data_row_num <= self.tot_data_rows:
                    self.data_row_num += 1
                    self.row_index = 0
                    self.clock.tick(7)
                self.state = self.STATE_WAITING

            # loads in previous row from file            
            if self.state == self.STATE_PREV:
                if self.data_row_num > self.data_start_row:
                    self.data_row_num -= 1
                    self.row_index = 0
                    self.clock.tick(7)
                self.state = self.STATE_WAITING

            # notes index for sub-trajectory demarcation                       
            if self.state == self.STATE_MARK:
                if self.first_index:
                    self.sub_index_list.append(self.data_row_num)
                    self.sub_index_list.append(self.row_index)
                    self.first_index = False
                    self.mpo_color = 1
                    self.clock.tick(7)
                else:
                    self.sub_index_list.append(self.row_index)
                    self.mpo_color = 0
                    self.first_index = True
                    self.clock.tick(7)
                self.state = self.STATE_PLAY

            # stops and saves sub-trajectory data
            if self.state == self.STATE_STOP:
                # create new training file, and append sub-trajectory indices
                subfile_name = self.file_name.split('\\')[-1].split('.')[0]
                new_name = 'train_' + subfile_name + '.csv'
                new_fname = os.path.join(os.path.split(self.file_name)[0],new_name)
                row_to_write = []
                self.ofile = open(new_fname,'wb')
                self.writer = csv.writer(self.ofile,delimiter=',')
                for i in range(0,len(self.sub_index_list)- 1,3):
                    row_to_write.append(self.sub_index_list[i])
                    row_to_write.append(self.sub_index_list[i+1])
                    row_to_write.append(self.sub_index_list[i+2])
                    self.writer.writerow(row_to_write)
                    row_to_write = []
                #clean and quit
                self.ofile.close()
                self.ifile.close()
                pygame.quit()
                break
            self.show_frame_info()
            pygame.display.flip()

if __name__ == "__main__":
    Viz_MPO().run()
