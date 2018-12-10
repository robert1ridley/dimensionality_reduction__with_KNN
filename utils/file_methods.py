def get_raw_data(filepath):
  infile = open(filepath, 'r')
  data = infile.readlines()
  infile.close()
  return data


def preprocess_data(in_data):
  list_of_lines = []
  for line in in_data:
    new_line = []
    line = line.split(',')
    for item in line:
      item = float(item.strip())
      new_line.append(item)
    list_of_lines.append(new_line)
  return list_of_lines
