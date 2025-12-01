from components.router import route_query

print(route_query("Summarize the uploaded PDF for me. \n"))
# Expected: decision='RAG'

print(route_query("Write a python script to scrape google. \n"))
# Expected: decision='DIRECT'

print(route_query("Who is the president of France in 2025? \n"))
# Expected: decision='WEB'
