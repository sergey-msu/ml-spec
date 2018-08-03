import time as t
import datetime as dt
import utils
#from course2_supervised_learning.week1 import header, run
#from course2_supervised_learning.week2 import header, #run
#from course2_supervised_learning.week3 import header, run
#from course2_supervised_learning.week4 import header, run
from course2_supervised_learning.week5 import header, run


def title():
    return 'COURSE 2: Supervised Learning';


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