import csv
import gtk
import pygame
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'viz_MPO.py'))
import viz_MPO

file_name = ''
ifile = ''
ofile = ''
reader = ''
writer = ''
lndmrk_row = '' # holds coordinates describing current landmark
tot_rows = 0 # total rows in file

tot_data_rows = 0 # number of mpo path coordinate rows
num_lndmarks = 0 # number of landmarks in file


#filechooser:
# uses gtk for file choosing dialog
def get_file_name():

    global tot_rows
    global file_name
    global ifile
    global reader
    global lndmrk_row
    
    dialog = gtk.FileChooserDialog("Open..",
                           None,
                           gtk.FILE_CHOOSER_ACTION_OPEN,
                           (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                            gtk.STOCK_OPEN, gtk.RESPONSE_OK))

    dialog.set_default_response(gtk.RESPONSE_OK)

    response = dialog.run()

    if response == gtk.RESPONSE_OK:
        tot_rows = len(open(dialog.get_filename()).readlines())
        file_name = dialog.get_filename()
        #ifile  = open(dialog.get_filename()) #move to init()
        #reader = csv.reader(ifile)#move to init()
        #lndmrk_row = reader.next()#move to init()
        dialog.destroy()
        
    elif response == gtk.RESPONSE_CANCEL:
        print 'Closed, no files selected'
        dialog.destroy()


def init():
    global file_name
    pass


    
# brute force row reading
def move_to_row(n_row):

    global ifile
    global reader
    global row
    
    ifile.close()
    ifile = open(file_name)
    reader = csv.reader(ifile)
    row = reader.next() # skip landmark row

    # load in desired row
    for i in range(0,n_row,1):
        row = reader.next()
    
# splits the chosen data file into many sequentially numbered
#   sub-files and runs subsequence extraction program on each.
def process_file():
    global tot_rows
    global file_name
    global ofile
    global writer
    global lndmrk_row
    
    # loop through data rows
    for i in range(1,tot_rows,1):

        # get name of datafile
        subfile_name = os.path.split(file_name)[1].split('.')[0]

        # append row sequence number
        new_name = subfile_name + str(i) + '.csv'
        
        # create new filename and open file.
        new_fname = os.path.join(os.path.split(file_name)[0], new_name)
        ofile = open(new_fname,'wb')
        writer = csv.writer(ofile,delimiter=',')

        # write landmark row
        writer.writerow(lndmrk_row)
        
        # write data row
        move_to_row(i)
        writer.writerow(row)
        ofile.close()

        # run sub-sequence extraction module
        extract = viz_MPO.Viz_MPO()
        extract.run(new_fname)
        del extract
        
def main():
    get_file_name()
    process_file()

main()
