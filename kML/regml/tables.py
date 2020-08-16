from typing import List


def generate_table(data: List[dict]):
    columns = data[0].keys()
    result = '<table class="table table-hover"><thead><tr><th scope="col">#</th>'
    for col in columns:
        result += f'<th>{col}</th>'
    result += '</tr></thead><tbody>'
    for idx, d in enumerate(data[:5]):
        new_row = f'<tr><th scope="row">{idx+1}</th>'
        for value in d.values():
            new_row += f'<td>{value}</td>'
        new_row += '</tr>'
        result += new_row
    result += '</tbody></table>'
    return result
