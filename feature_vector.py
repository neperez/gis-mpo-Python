from math import *
from random import *
from sklearn import svm
from sklearn import cross_validation
import pylab
import numpy
import csv
import gtk
import os

class MPO:

    feature_types = ['distance','speed','accel','anglev','anglevD','anglevDD','angleh','anglehD','anglehDD']
    
    
    def __init__(self, identifier, trajectory):
        """ Initializes the MPO object. The argument named identifier is a tuple
        (filename, rownumber) which uniquely identifies where the MPO data came
        from. This will saved in an instance variable and may later be useful
        for housekeeping purposes. It initializes the points on the MPO trajectory.
        It also initializes the point list on the
        reference landmark to the empty list [ ] - it is meant to be set by
        the setReferenceLandmark method below.
        """
        self.trajectory = trajectory
        self.identifier = identifier
        self.init_data()

    def init_data(self):
        self.lndmrk_pts = []
        self.distance_FVs = []
        self.speed_FVs = []
        self.accel_FVs = []
        self.anglev_FVs = []
        self.anglevD_FVs = []
        self.anglevDD_FVs = []
        self.angleh_FVs = []
        self.anglehD_FVs = []
        self.anglehDD_FVs = []
        
    def __repr__(self):
        mystr = '[ '
        for pt in self.trajectory:
            mystr += '<' + str(pt[0]) + ',' + str(pt[1]) + '> '
        mystr += ']'
        return mystr
        
    def setReferenceLandmark(self,periphery):
        """ Sets the reference landmark for the MPO. The other methods will
        compute values like distance, speed with reference to this landmark.
        Typical usage is - first set the landmark, then call the get methods.
        To use another landmark, call set again on the new landmark.
        Argument periphery is the list of points on the periphery of landmark
        which is a general polygonal shape. A point is a tuple like (x,y) and
        the periphery is like [ (x1,y1), (x2,y2), ... , (xn,yn) ]
        One the new landmark is set all the feature vectors are updated.
        """
        self.init_data()
        self.lndmrk_pts = periphery
        n_lndmrk = len(self.lndmrk_pts) # number of points on landmark
        n_trj = len(self.trajectory) # number of points on trajectory
        
        # create distance feature vectors
        for pt in self.trajectory:
            distance_FV = []
            for lndmrk_pt in self.lndmrk_pts:
                
                distance = sqrt( ( float(pt[0]) - lndmrk_pt[0] )**2 + ( float(pt[1]) - lndmrk_pt[1] )**2 )
                distance_FV.append( distance )
            self.distance_FVs.append(distance_FV)
        
        # create speed feature vectors
        for index in range(n_trj):
            speed_FV = []

            if index < (n_trj-1):
                distance_FV_next = self.distance_FVs[index + 1]
                distance_FV_prev = self.distance_FVs[index]
            else:
                distance_FV_next = self.distance_FVs[index]
                distance_FV_prev = self.distance_FVs[index - 1]
                
            for i in range(n_lndmrk):
                speed_FV.append( distance_FV_next[i] - distance_FV_prev[i] )
                
            self.speed_FVs.append(speed_FV)

        # create acceleration feature vectors
        for index in range(n_trj):
            accel_FV = []

            if index < (n_trj-1):
                speed_FV_next = self.speed_FVs[index + 1]
                speed_FV_prev = self.speed_FVs[index]
            else:
                speed_FV_next = self.speed_FVs[index]
                speed_FV_prev = self.speed_FVs[index - 1]
                
            for i in range(n_lndmrk):
                accel_FV.append( speed_FV_next[i] - speed_FV_prev[i] )
                
            self.accel_FVs.append(accel_FV)

        # create anglev (angle measured with respect to a vertical line on the plane) feature vectors
        for pt in self.trajectory:
            anglev_FV = []
            for lndmrk_pt in self.lndmrk_pts:
                v = [ ( float(pt[0]) - lndmrk_pt[0] ), ( float(pt[1]) - lndmrk_pt[1] ) ] # the geometric vector pointing from landmark point to MPO location
                anglev = getAngle(v)
                anglev_FV.append( anglev )
            self.anglev_FVs.append(anglev_FV)

        # create anglevD feature vectors
        for index in range(n_trj):
            anglevD_FV = []

            if index < (n_trj-1):
                anglev_FV_next = self.anglev_FVs[index + 1]
                anglev_FV_prev = self.anglev_FVs[index]
            else:
                anglev_FV_next = self.anglev_FVs[index]
                anglev_FV_prev = self.anglev_FVs[index - 1]
                
            for i in range(n_lndmrk):
                anglevD_FV.append( anglev_FV_next[i] - anglev_FV_prev[i] )
                
            self.anglevD_FVs.append(anglevD_FV)

        # create anglevDD feature vectors
        for index in range(n_trj):
            anglevDD_FV = []

            if index < (n_trj-1):
                anglevD_FV_next = self.anglevD_FVs[index + 1]
                anglevD_FV_prev = self.anglevD_FVs[index]
            else:
                anglevD_FV_next = self.anglevD_FVs[index]
                anglevD_FV_prev = self.anglevD_FVs[index - 1]
                
            for i in range(n_lndmrk):
                anglevDD_FV.append( anglevD_FV_next[i] - anglevD_FV_prev[i] )
                
            self.anglevDD_FVs.append(anglevDD_FV)

        # create angleh (angle measured with respect to a horizontal line on the plane) feature vectors
        for pt in self.trajectory:
            angleh_FV = []
            for lndmrk_pt in self.lndmrk_pts:
                v = [ ( float(pt[0]) - lndmrk_pt[0] ), ( float(pt[1]) - lndmrk_pt[1] ) ] # the geometric vector pointing from landmark point to MPO location
                angleh = getAngle(v,[1,0])
                angleh_FV.append( angleh )
            self.angleh_FVs.append(angleh_FV)

        # create anglehD feature vectors
        for index in range(n_trj):
            anglehD_FV = []

            if index < (n_trj-1):
                angleh_FV_next = self.angleh_FVs[index + 1]
                angleh_FV_prev = self.angleh_FVs[index]
            else:
                angleh_FV_next = self.angleh_FVs[index]
                angleh_FV_prev = self.angleh_FVs[index - 1]
                
            for i in range(n_lndmrk):
                anglehD_FV.append( angleh_FV_next[i] - angleh_FV_prev[i] )
                
            self.anglehD_FVs.append(anglehD_FV)

        # create anglehDD feature vectors
        for index in range(n_trj):
            anglehDD_FV = []

            if index < (n_trj-1):
                anglehD_FV_next = self.anglehD_FVs[index + 1]
                anglehD_FV_prev = self.anglehD_FVs[index]
            else:
                anglehD_FV_next = self.anglehD_FVs[index]
                anglehD_FV_prev = self.anglehD_FVs[index - 1]
                
            for i in range(n_lndmrk):
                anglehDD_FV.append( anglehD_FV_next[i] - anglehD_FV_prev[i] )
                
            self.anglehDD_FVs.append(anglehDD_FV)

        
    def getFVs(self, Fname, begin, end):
        """ Gets a list of feature vectors. Argument Fname is given the
        feature type name: distance, speed, accel, angle, angleD, or angleDD.
        The indices begin and end are used to specify the sub-trajectory
        of the MPO for which this is being done.
        """
        if Fname=='distance': return self.distance_FVs[ begin : end + 1]
        elif Fname=='speed': return self.speed_FVs[ begin : end + 1]
        elif Fname=='accel': return self.accel_FVs[ begin : end + 1]
        elif Fname=='anglev': return self.anglev_FVs[ begin : end + 1]
        elif Fname=='anglevD': return self.anglevD_FVs[ begin : end + 1]
        elif Fname=='anglevDD': return self.anglevDD_FVs[ begin : end + 1]
        elif Fname=='angleh': return self.angleh_FVs[ begin : end + 1]
        elif Fname=='anglehD': return self.anglehD_FVs[ begin : end + 1]
        elif Fname=='anglehDD': return self.anglehDD_FVs[ begin : end + 1]
    

    def get_avg_FV(self, Fname, begin, end):
        """ Gets the average feature vector. Argument Fname is given the
        feature type name: distance, speed, accel, angle, angleD, or angleDD.
        The indices begin and end are used to specify the sub-trajectory
        of the MPO for which the average is being computed.
        """
        FVs = self.getFVs(Fname, begin, end)
        n_lndmrk = len(FVs[0]) # number of measurements in a single FV which is also number of points on a landmark
        FV_avg = n_lndmrk * [0]
        for i in range(end-begin+1):
            for j in range(n_lndmrk):
                FV_avg[j] = FV_avg[j] + FVs[i][j]
        for k in range(n_lndmrk):
            FV_avg[k] = FV_avg[k] / (end - begin + 1)

        return FV_avg
    
    def get_avg_full_feature_vector(self, begin, end):
        avg_FV = []
        for ftype in MPO.feature_types:    
            avg_FV += self.get_avg_FV(ftype,begin,end)
        return avg_FV

# ENDS MPO CLASS

# HELPER FUNCTIONS

def getAngle(v,v0=[0,-1]):
    """ v0 is the reference vector
    """
    return acos( (v[0]*v0[0] + v[1]*v0[1]) / sqrt( v[0]**2 + v[1]**2 ) )

def writeToCSV(filename, mpo_list, Fname, index):
    """ Fname is feature name, index is the specific index within
    the feature vector that is being written in the file. In other
    words index corresponds to a point on the landmark.
    """
    wrtr = csv.writer( open(filename, 'wb'), delimiter=',')
    for mpo in mpo_list:
        n_trj = len(mpo.trajectory)
        fvs = mpo.getFVs(Fname, 0, n_trj-1)
        data = []
        for fv in fvs:
            data.append(fv[index])
        wrtr.writerow(data)

def readFromCSV(filename):
    rdr = csv.reader( open(filename, 'rb'), delimiter=',')
    mpo_dict = dict()
    landmark_dict = dict()
    for i,row in enumerate(rdr):
        if i==0: # the header row which has number of landmarks on 4th column
            n_lndmrks = int(row[3])
        elif i <= n_lndmrks: # must be a row which has landmark points
            landmark_dict[i] = seq2pointList(row)
        else: # must be a row which has mpo points
            mpo_dict[i] = MPO( (filename,i), seq2pointList(row) ) 
    return [ landmark_dict, mpo_dict ]

def seq2pointList(seq):
    """ seq is a list of strings which will be
    converted to list of (X,Y) pairs / points
    """
    pointList = []
    for i in range(0,len(seq)-1,2):
        if seq[i] != '': # csv files are padded with empty elements
            pointList.append( ( float(seq[i]), float(seq[i+1]) ) )
    return pointList

def get_file_number():
    """ Uses gtk to prompt the user for the number of files to process """
    dialog = gtk.MessageDialog(None,
                               gtk.DIALOG_MODAL | gtk.DIALOG_DESTROY_WITH_PARENT,
                               gtk.MESSAGE_QUESTION,
                               gtk.BUTTONS_OK,
                               None)
    dialog.set_markup('Enter the number of')
    entry = gtk.Entry()
    entry.connect('activate', responseToDialog, dialog, gtk.RESPONSE_OK)
    hbox= gtk.HBox()
    hbox.pack_start(gtk.Label('Enter:'), False, 5, 5)
    hbox.pack_end(entry)
    dialog.format_secondary_markup('data files you will be using')
    dialog.vbox.pack_end(hbox, True, True,0)
    dialog.show_all()
    dialog.run()
    text = entry.get_text()
    dialog.destroy()
    return text

def responseToDialog(entry, dialog, response):
    """ helper function for the file number prompt """
    dialog.response(response)

def get_file_path(file_number):
    """ uses gtk to get file paths from user """
    dialog = gtk.FileChooserDialog("Open file number " + str(file_number),
                           None,
                           gtk.FILE_CHOOSER_ACTION_OPEN,
                           (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                            gtk.STOCK_OPEN, gtk.RESPONSE_OK))

    dialog.set_default_response(gtk.RESPONSE_OK)

    response = dialog.run()

    if response == gtk.RESPONSE_OK:
        file_name = dialog.get_filename()
        dialog.destroy()
        return file_name
    elif response == gtk.RESPONSE_CANCEL:
        print 'Closed, no files selected'
        dialog.destroy()

def process_data():
    """ fetches sub-trajectories from files and creates
    mpo feature vectors """
    
    raw_file_paths = [] # collection of file paths of raw data files
    ''' raw_file data layout:
    [ [ [landmark Vertices set #1]...[landmark vertices set #n ] ] [ [mpo path list #1]...[mpo path list #n] ] ]
    '''
    class_names =[]

    mpo_fv_list = []
    target_classes = []

    raw_file_data = [] # collection of list tuples of landmark vertices and mpo paths
    
    train_prefix = 'train_' # prefix for file path building
      
    # gtk prompt for number of data files
    num_of_files = get_file_number()

    #  read in file paths for chosen number of files and save file paths to list
    for i in range(int(num_of_files)):
        raw_file_paths.append(get_file_path(i+1))
    
    # for each file path in list,
    for i in range(len(raw_file_paths)):
        
        # use readFromCSV into a collection list
        [ landmark_dict, mpo_dict ] = readFromCSV(raw_file_paths[i])

        landmark_pts = landmark_dict[1] # assuming there is one landmark only
        
        # save respective training file path
        split_path = os.path.split(raw_file_paths[i])
        class_name = split_path[1].split('.')[0]
        class_names.append(class_name)
        new_path_suffix = train_prefix + split_path[1]
        train_file_path = os.path.join(split_path[0], new_path_suffix)
        
        # open training file:
        train_file = open(train_file_path,'rb')
        train_reader = csv.reader(train_file)
                
        
        # for each row in training file:
        for row in train_reader:
            # gather traversal info
            data_row_num = int(row[0])
            begin = int(row[1])/2
            end = int(row[2])/2
            mpo = mpo_dict[data_row_num]
            mpo.setReferenceLandmark(landmark_pts)
            mpo_fv_list.append( mpo.get_avg_full_feature_vector(begin,end) )
            target_classes.append(i)

    print(target_classes)
    print(class_names)
    
    # create SVM
    clf = svm.SVC()

    # train & test number 1
    train_data = mpo_fv_list[0:16] + mpo_fv_list[20:36] + mpo_fv_list[40:56]
    train_classes = target_classes[0:16] + target_classes[20:36] + target_classes[40:56]
    clf.fit(train_data,train_classes)
    test_data = mpo_fv_list[16:20] + mpo_fv_list[36:40] + mpo_fv_list[56:60]
    true_classes = target_classes[16:20] + target_classes[36:40] + target_classes[56:60]
    print("\ntest results from trained SVM - ")
    predicted_classes = clf.predict(test_data)
    print( predicted_classes )
    common_elements = [i for i, j in zip(predicted_classes, true_classes) if i == j]
    error = 100 * ( 1 - len(common_elements) / float(len(predicted_classes)) )
    print("\npercent error in prediction = %f\n" %error)

#process_data()

# TEST FUNCTIONS

def test1():
    mpo1 = MPO( ("does not matter in this case","does not matter in this case"), [ (0,0), (1,1), (2,2), (3,3), (4,4), (5,5) ] )
    mpo1.setReferenceLandmark( [ (0,3), (0,4), (1,4), (1,3) ] )
    print("\nOUTPUTS FROM TEST 1:")
    begin, end = 1, 4
    for ftype in MPO.feature_types:    
        print("\n%s FVs between indices %d & %d - " %(ftype,begin,end))
        print ( mpo1.getFVs(ftype,begin,end) )
        print("\naverage %s FV between indices %d & %d - " %(ftype,begin,end))
        print ( mpo1.get_avg_FV(ftype,begin,end) )

    print("\nfull (averaged) feature vector between indices %d & %d - " %(begin,end))
    print ( mpo1.get_avg_full_feature_vector(begin,end) )
    
#test1()

def test2():
    mpo1 = MPO( ("does not matter in this case","does not matter in this case"), [ (0,0), (1,1), (2,2), (3,3), (4,4), (5,5) ] )
    mpo1.setReferenceLandmark( [ (3,0), (3,1), (4,1), (4,0) ] )
    print("\nOUTPUTS FROM TEST 2:")
    begin, end = 1, 4
    for ftype in MPO.feature_types:    
        print("\n%s FVs between indices %d & %d - " %(ftype,begin,end))
        print ( mpo1.getFVs(ftype,begin,end) )
        print("\naverage %s FV between indices %d & %d - " %(ftype,begin,end))
        print ( mpo1.get_avg_FV(ftype,begin,end) )

    print("\nfull (averaged) feature vector between indices %d & %d - " %(begin,end))
    print ( mpo1.get_avg_full_feature_vector(begin,end) )
    
#test2()

def test3():
    [ landmark_list, mpo_list ] = readFromCSV('..\\pattern_creator\\data2.csv')
    for mpo in mpo_list:
        mpo.setReferenceLandmark( landmark_list[0])
    writeToCSV( 'distance_samples_data2_0.csv',mpo_list, MPO.feature_types[0], 0 )
    
#test3()

def getNoisyPoints(pts,level):
    """ Helper function for svm train test intended to quickly generate similar datasets with small variations.
    The argument level is the level of noise used to multiply the 0-1 fraction returned by random( )
    """
    noisy_pts = []
    for pt in pts:
        noisy_pts.append( (pt[0] + level * random(), pt[1] + level * random()) )
    return noisy_pts

def create_test_files(noiseLevel):
    landmark_points = [ (0,3), (0,4), (1,4), (1,3) ]
    type1_pts = [ (0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6) ]
    type2_pts = [ (0,0), (-1,1), (-2,2), (-3,3), (-4,4), (-5,5), (-6,6) ]
    type3_pts = [ (0,5), (-1,5), (-2,5), (-3,5), (-4,5), (-5,5), (-6,5) ]
    #noiseLevel = 0.5
    for i in range(1,4):
        filename = 'type'+str(i)+'.csv'
        wrtr = csv.writer( open(filename, 'wb'), delimiter=',')
        wrtr.writerow(['No image',600,600,1])
        wrtr.writerow([0,3,0,4,1,4,1,3])
        twrtr = csv.writer( open('train_'+filename, 'wb'), delimiter=',')
        pts_list = eval('type'+str(i)+'_pts')
        for i in range(20):
            pts = getNoisyPoints(pts_list,noiseLevel)
            coords = []
            for pt in pts:
                coords.append(pt[0])
                coords.append(pt[1])
            wrtr.writerow(coords)
            train_data = [i+2, choice([0,2]), choice([len(coords)-4,len(coords)-2]) ]
            twrtr.writerow(train_data)
        
    
def svm_train_test1():
    landmark_points = [ (0,3), (0,4), (1,4), (1,3) ]
    type1_pts = [ (0,0), (1,1), (2,2), (3,3), (4,4), (5,5) ]
    type2_pts = [ (0,0), (-1,1), (-2,2), (-3,3), (-4,4), (-5,5) ]
    #type3_pts = [ (2,5), (1,5), (0,5), (-1,5), (-2,5), (-3,5), (-4,5), (-5,5) ]
    type3_pts = [ (0,5), (-1,5), (-2,5), (-3,5), (-4,5), (-5,5) ]
    #type3_pts = [ (0,100), (-1,100), (-2,100), (-3,100), (-4,100), (-5,100) ]

    mpo_type1_list = []
    mpo_type2_list = []
    mpo_type3_list = []
    mpo_type1_FVs = []
    mpo_type2_FVs = []
    mpo_type3_FVs = []
    begin,end = 0,5
    noiseLevel = 0.5

    for i in range(20):
        mpo = MPO( ("",""), getNoisyPoints(type1_pts,noiseLevel) )
        mpo.setReferenceLandmark( landmark_points )
        mpo_type1_list.append( mpo )
        mpo_type1_FVs.append( mpo.get_avg_full_feature_vector(begin,end) )

        mpo = MPO( ("",""), getNoisyPoints(type2_pts,noiseLevel) )
        mpo.setReferenceLandmark( landmark_points )
        mpo_type2_list.append( mpo )
        mpo_type2_FVs.append( mpo.get_avg_full_feature_vector(begin,end) )

        mpo = MPO( ("",""), getNoisyPoints(type3_pts,noiseLevel) )
        mpo.setReferenceLandmark( landmark_points )
        mpo_type3_list.append( mpo )
        mpo_type3_FVs.append( mpo.get_avg_full_feature_vector(begin,end) )
        
    # create SVM
    clf = svm.SVC()
##    mpo_fv_list = mpo_type1_FVs + mpo_type2_FVs + mpo_type3_FVs
##    target_classes = 20 * [1] + 20 * [2] + 20 * [3]
##    scores = cross_validation.cross_val_score(clf, mpo_fv_list, target_classes, cv=5)
##    print(scores)
    
    # train with some type1, type2 & type3 MPOs
    classes = 16 * [1] + 16 * [2] + 16 * [3] 
    data = mpo_type1_FVs[0:16] + mpo_type2_FVs[0:16] + mpo_type3_FVs[0:16]
    clf.fit(data,classes)
    # test with some type1, type2 & type3 MPOs
    test_data = mpo_type1_FVs[16:] + mpo_type2_FVs[16:] + mpo_type3_FVs[16:]
    true_classes = 4 * [1] + 4 * [2] + 4 * [3]
    print("\ntest results from trained SVM - ")
    predicted_classes = clf.predict(test_data)
    print( predicted_classes )
    common_elements = [i for i, j in zip(predicted_classes, true_classes) if i == j]
    error = 100 * ( 1 - len(common_elements) / float(len(predicted_classes)) )
    print("\npercent error in prediction = %f\n" %error)
##
##    n = 1 # number of MPOs to write data from each type
##    writeToCSV( 'distance_samples_0.csv',mpo_type1_list[:n] + mpo_type2_list[:n] + mpo_type3_list[:n], MPO.feature_types[0], 0 ) # take n mpos from each list and select distance feature only and write only index 0 value
##    writeToCSV( 'anglev_samples_0.csv',mpo_type1_list[:n] + mpo_type2_list[:n] + mpo_type3_list[:n], MPO.feature_types[3], 0 ) # for anglev feature
##
##    writeToCSV( 'distance_samples_2.csv',mpo_type1_list[:n] + mpo_type2_list[:n] + mpo_type3_list[:n], MPO.feature_types[0], 2 ) 
##    writeToCSV( 'anglev_samples_2.csv',mpo_type1_list[:n] + mpo_type2_list[:n] + mpo_type3_list[:n], MPO.feature_types[3], 2 ) 
 
#svm_train_test1()

