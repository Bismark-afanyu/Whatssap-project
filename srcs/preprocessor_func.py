import re

class MessageParser:
    @staticmethod
    def startsWithDateAndTimeAndroid(s):
        pattern = r'''
            (?:\[?(?P<Date>\d{2}[/-]\d{2}[/-]\d{2,4}),?\s+(?P<Time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]?\s*-?\s*)?  # Date and time (optional)
            (?:~?\s*\u202f?\s*(?P<Sender>[^:]+):\s*)?  # Sender (optional)
            (?P<Message>.+)  # Message (required)
        '''  
        regex = re.compile(pattern, re.VERBOSE)
        result = regex.match(s)
        return bool(result)  # Returns True if match is found, False otherwise

    @staticmethod
    def FindAuthor(s):
        if ':' in s:
            return s.split(':', 1)[0].strip()  # Return the author's name
        return None  # Return None if no author is found

    @staticmethod
    def getDataPointAndroid(line):
        splitline = line.split(' - ', 1)  # Split by ' - ' to separate datetime from the rest
        if len(splitline) < 2:
            return None, None, None, None  # Handle cases where the line doesn't match the expected format
        
        dateTime = splitline[0]
        message = splitline[1] if len(splitline) > 1 else ""  # Ensure message is initialized

        # Debug: Print the dateTime string to verify its format
        print(f"dateTime: {dateTime}")

        # Split date and time
        if ', ' in dateTime:
            date, time = dateTime.split(', ')
        else:
            # Handle cases where the dateTime format is unexpected
            date, time = dateTime, ""  # Assign dateTime to date and leave time as empty

        # Extract author if present
        author = None
        if MessageParser.FindAuthor(message):  # Check if message contains an author
            splitMessage = message.split(':', 1)  # Split by ':' to separate author and message
            author = splitMessage[0].strip()
            message = splitMessage[1].strip() if len(splitMessage) > 1 else ""

        return date, time, author, message

    @staticmethod
    def startsWithDateAndTimeios(line):
        # Check if the line starts with a date-time pattern
        if not line.startswith('[') or ']' not in line:
            return None, None, None, line  # Return the entire line as the message if no date-time is found

        # Split the line into dateTime and the rest
        splitLine = line.split('] ', 1)  # Split on the first '] '
        if len(splitLine) < 2:
            return None, None, None, line  # Handle cases where the line doesn't contain a message

        dateTime, message = splitLine
        dateTime = dateTime[1:]  # Remove the leading '['

        # Split dateTime into date and time
        if ',' in dateTime:
            date, time = dateTime.split(',', 1)  # Split on the first comma
        elif ' ' in dateTime:
            date, time = dateTime.split(' ', 1)  # Split on the first space
        else:
            # If no separator is found, assume the entire string is the date
            date, time = dateTime, ''

        # Clean up the time format
        if 'AM' in time or 'PM' in time:
            # Handle 12-hour format (e.g., "2:30:00 PM")
            time = time.strip()  # Remove leading/trailing spaces
        else:
            # Handle 24-hour format (e.g., "14:30:00")
            time = time.strip()

        # Extract author and message
        author = None
        if MessageParser.FindAuthor(message):  # Assuming FindAuthor is a function to check if the message contains an author
            splitMessage = message.split(':', 1)  # Split on the first ':'
            if len(splitMessage) > 1:
                author, message = splitMessage
                author = author.strip()  # Remove leading/trailing spaces
                message = message.strip()

        return date, time, author, message
