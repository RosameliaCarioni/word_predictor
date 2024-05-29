def read_lines_from_file_as_data_chunks(file_name, chunk_size, callback, return_whole_chunk=False):
    """
    https://gist.github.com/iyvinjose/e6c1cb2821abd5f01fd1b9065cbc759d 
    Read file line by line regardless of its size.
    :param file_name: absolute path of file to read
    :param chunk_size: size of data to be read at a time
    :param callback: callback method, prototype ----> def callback(data, eof, file_name)
    """

    def read_in_chunks(file_obj, chunk_size=5000):
        """
        Lazy function to read a file in chunks.
        Default chunk size: 5000.
        """
        while True:
            data = file_obj.read(chunk_size)
            if not data:
                break
            yield data

    with open(file_name, 'r', encoding='utf-8') as fp:
        data_left_over = None

        # Loop through characters
        for chunk in read_in_chunks(fp, chunk_size):

            # If uncompleted data exists
            if data_left_over:
                current_chunk = data_left_over + chunk
            else:
                current_chunk = chunk

            # Split chunk by new line
            lines = current_chunk.splitlines()

            # Check if line is complete
            if current_chunk.endswith('\n'):
                data_left_over = None
            else:
                data_left_over = lines.pop()

            if return_whole_chunk:
                callback(data=lines, eof=False, file_name=file_name)
            else:
                for line in lines:
                    callback(data=line, eof=False, file_name=file_name)

        if data_left_over:
            lines = data_left_over.splitlines()
            if return_whole_chunk:
                callback(data=lines, eof=False, file_name=file_name)
            else:
                for line in lines:
                    callback(data=line, eof=False, file_name=file_name)

        callback(data=None, eof=True, file_name=file_name)
