import os

class PATH:
    CURRENT   = os.path.dirname(os.path.realpath(__file__))
    RESOURCES = os.path.join(CURRENT, '..\materials')
    STORE     = os.path.join(CURRENT, '..\store')

    @staticmethod
    def COURSE_PATH(course):
        return os.path.join(PATH.RESOURCES, 'course'+str(course))

    @staticmethod
    def COURSE_FILE(course, fn, dir=None):
        path = PATH.COURSE_PATH(course)
        root = path if dir is None else os.path.join(path, dir)
        return os.path.join(root, fn)

    @staticmethod
    def STORE_FOR(course, fn, dir=None):
        root = os.path.join(PATH.STORE, 'course'+str(course))
        dpath = root if dir is None else os.path.join(root, dir)
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        return os.path.join(dpath, fn)

    @staticmethod
    def SAVE_RESULT(course_week, major_minor, obj):
        dpath = PATH.STORE_FOR(course_week[0],
                               dir='week'+str(course_week[1]),
                               fn='d_{0}_{1}.txt'.format(major_minor[0], major_minor[1]))
        result = ' '.join(str(x) for x in obj) if isinstance(obj, list) else str(obj)
        with open(dpath, 'w') as file:
            file.write(result)
        return

class PRINT:
  @staticmethod
  def HEADER(text, len=50, begin_line=True, end_line=True):
      template = '{:*^'+str(len)+'}'
      if begin_line: template = '\n'+template
      if end_line:   template = template + '\n'
      print(template.format(' '+text+' '))
