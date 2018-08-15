import time as t
import datetime as dt
import utils
#from course5_applications.week1 import header, run
#from course5_applications.week2 import header, run
#from course5_applications.week3 import header, run
from course5_applications.week4 import header, run


def title():
    return 'COURSE 5: Data Analysis Applications';


def main(args=None):

  utils.PRINT.HEADER(header())
  print('STARTED ', dt.datetime.now())
  start = t.time()
  run()
  end = t.time()
  utils.PRINT.HEADER('DONE in {}s ({}m)'.format(round(end-start, 2), round((end-start)/60, 2)))

  return


if __name__=='__main__':
  main()