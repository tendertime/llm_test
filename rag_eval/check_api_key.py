import os
api_key = os.environ.get('OPENAI_API_KEY')
if api_key:
    print('OPENAI_API_KEY: SET (length:', len(api_key), 'chars)')
else:
    print('OPENAI_API_KEY: NOT SET')
    print('\nPlease set your API key:')
    print('set OPENAI_API_KEY=your_key_here')
