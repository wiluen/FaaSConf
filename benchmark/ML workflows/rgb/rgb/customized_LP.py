from typing import Optional
from line_profiler import LineProfiler
from functools import wraps
import inspect
from datetime import datetime
import linecache
import pickle
import logging
import sys
class MyLineProfiler:
    def __init__(self):
        self.functions: list[list] = []
        self.line_profiler: Optional[LineProfiler] = None

    def __call__(self, func):
        index = len(self.functions)

        @wraps(func)
        def wrap(*args, **kw):
            return self.functions[index][1](*args, **kw)

        self.functions.append([func, func])
        return wrap

    def save_profiling(self, stream=None, output_unit=None, stripzeros=False):
        """
            save the profiling to an object
        """
        lstats = self.line_profiler.get_stats()
        re = self.show_text(lstats.timings, lstats.unit, output_unit=output_unit, stream=stream, stripzeros=stripzeros)
        return re
    def show_func_cust(self, filename, start_lineno, func_name, timings, unit,
              output_unit=None, stream=None, stripzeros=False):
#        print(filename)
        template = '%6s %9s %12s %8s %8s  %-s'
        d = {}
        total_time = 0.0
        linenos = []
        for lineno, nhits, time in timings:
            total_time += time
            linenos.append(lineno)
        if stripzeros and total_time == 0:
            return

        if output_unit is None:
            output_unit = unit
        scalar = unit / output_unit
        # total_time = total_time * unit
        # function info: func_name 
            # function line info: start_lineno
        #print("total time:\n")
        #print(total_time*unit)
        #print(f'File: {filename}\n')
        #print(f'Function: {func_name} at line {start_lineno}\n')
        # stream.write('Total time: %g s\n' % (total_time * unit))
        all_lines = linecache.getlines(filename)
        sublines = inspect.getblock(all_lines[start_lineno - 1:])
        result = []
        for lineno, nhits, time in timings:
            d[lineno] = (nhits,
                        '%5.1f' % (time * scalar))
        # range of line numbers
        linenos = range(start_lineno, start_lineno + len(sublines))
        empty = ('', '')
        for lineno, line in zip(linenos, sublines):
            nhits, time= d.get(lineno, empty)
            result.append([lineno]+ [nhits, time,
                            line.rstrip('\n').rstrip('\r')])
        return result
    def show_text(self, stats, unit, output_unit=None, stream=None, stripzeros=False):
#        if output_unit is not None:
#            print('Timer unit: %g s\n\n' % output_unit)
#        else:
#            print('Timer unit: %g s\n\n' % unit)
        result = []
        for (fn, lineno, name), timings in sorted(stats.items()):
            result.append(self.show_func_cust(fn, lineno, name, stats[fn, lineno, name], unit,
                    output_unit=output_unit, stream=stream,
                    stripzeros=stripzeros))
        return result

    def start(self):
        self.line_profiler = LineProfiler()
        for f in self.functions:
            f[1] = self.line_profiler(f[0])
            
    def stop(self, *, print: bool = True):
        for f in self.functions:
            f[1] = f[0]
        if self.line_profiler and print:
            re = self.save_profiling()
            #logging.info(re[0])
            result = ''.join(str(l) for l in re[0])
            sys.stderr.write(result)
            # timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            # file_name = timestr + ".obj"
            # file = open('logdata/profile_u5/'+file_name,'wb')
            # pickle.dump(re, file)
            # file.close()
#        self.line_profiler.print_stats()
    def reset(self):
        self.stop(print=False)
        self.start()
