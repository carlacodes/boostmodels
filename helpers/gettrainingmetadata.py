from pathlib import Path
import scipy
import scipy.io
import os
import datetime
import shutil


def get_training_metadata(ferrets):
    list_of_dicts = []

    for ferret in ferrets:
        path = Path('D:\Data\L27and28_training/' + ferret + '/')
        #extract the matlab files from the path folder
        path_list = [path for path in path.glob('**/*.mat')]
        list_of_dicts_for_ferret = []
        list_of_dicts_training = []
        #load the matlab files
        for path in path_list:
            #load the matlab file
            mat = scipy.io.loadmat(path)
            #get the name of the matlab file
            filename = os.path.basename(path)
            #get the name of the matlab file without the extension
            filename = os.path.splitext(filename)[0]
            #get the name of the folder that the matlab file is in
            foldername = os.path.basename(os.path.dirname(path))
            #get the name of the folder that the matlab file is in without the extension
            foldername = os.path.splitext(foldername)[0]
            #get the date of the matlab file
            day = filename.split('_')[0]
            month = filename.split('_')[1]
            year = filename.split('_')[2]
            year = year[0:4]
            #convert the date to a datetime object
            date = datetime.datetime(int(year), int(month), int(day))
            #append to a list of dictionaries
            list_of_dicts.append({'filename': filename, 'date': date, 'ferret': ferret})
            list_of_dicts_for_ferret.append({'filename': filename, 'date': date, 'ferret': ferret})
        #sort the list of dictionaries by date
        list_of_dicts_for_ferret.sort(key=lambda x: x['date'])
        #get the first and last date
        first_date = list_of_dicts_for_ferret[1]['date'] #take the second date because the first date likely could be an error
        last_date = list_of_dicts_for_ferret[-1]['date']
        #get the number of days between the first and last date
        num_days = (last_date - first_date).days
        #get the number of sessions
        #get the first three weeks of sessions
        #filter out the first three weeks of sessions
        list_of_dicts_for_ferret2 = [x for x in list_of_dicts_for_ferret if (x['date'] - first_date).days <= 21]
        #move each file in list_of_dicts_for_ferret2 to a different directory
        for file in list_of_dicts_for_ferret2:
            #get the name of the file
            filename = file['filename']
            #get the name of the folder that the file is in
            foldername = file['ferret']
            #get the path of the file
            path = Path('D:\Data\L27and28_training/' + foldername + '/' + filename + '.mat')
            #get the path of the new directory
            new_path = Path('D:\Data\L27and28_training_filtered/' + foldername + '/' + filename + '.mat')
            #create the new directory
            os.makedirs(new_path.parent, exist_ok=True)
            if os.path.exists(path):
                # Copy the file to the new directory
                shutil.copy(path, new_path)
                print(f"File '{filename}.mat' copied to '{new_path}'")
            else:
                print(f"File '{filename}.mat' does not exist or the path is incorrect.")


        num_sessions = len(list_of_dicts_for_ferret)

    return list_of_dicts

def main():
    ferrets = ['F1702_Zola', 'F1815_Cruella', 'F1803_Tina', 'F2002_Macaroni', 'F2105_Clove']  # , 'F2105_Clove']
    metadata = get_training_metadata(ferrets)
    #
    # for ferret in ferrets:
    #     run_correctrxntime_model_for_a_ferret([ferret], optimization=False, ferret_as_feature=False)



if __name__ == '__main__':
    main()