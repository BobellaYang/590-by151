#NOTE: paste in linux is usually cntl+shift+v (not cntl+v)
ScriptLoc=${PWD} #save script directory path as shell variable
for i in *.py; do echo $i; python $i; done #run all python scripts in directory

cd LectureCodes #return to script directory
for i in *.py; do echo $i; python $i; done #run all python scripts in directory
grep 'I HAVE WORKED' *
#NOTE: grep searches through files for matching strings (very useful command)