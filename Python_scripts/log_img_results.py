import HTML


def Log_results_in_text_file()
HTMLFILE = 'HTML_tutorial_output.html'
f = open(HTMLFILE, 'w')


#=== TABLES ===================================================================

# 1) a simple HTML table may be built from a list of lists:

table_data = [
        ['Last name',   'First name',   'Age'],
        ['Smith',       'John',         30],
        ['Carpenter',   'Jack',         47],
        ['Johnson',     'Paul',         62],
    ]

htmlcode = HTML.table(table_data)
print htmlcode
f.write(htmlcode)
f.write('<p>')
print '-'*79

