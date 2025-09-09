#!/bin/bash

echo "Testing GoScriptureAPI"
echo "====================="

# Test health endpoint
echo -e "\n1. Testing Health Endpoint:"
curl -s http://localhost:8080/health | jq .

# Test status endpoint
echo -e "\n2. Testing Status Endpoint:"
curl -s http://localhost:8080/status | jq .

# Test search endpoint with GET request
echo -e "\n3. Testing Search Endpoint (GET request):"
curl -s "http://localhost:8081/search?q=For%20God%20so%20loved%20the%20world&k=3" | jq .

# Test search with POST request
echo -e "\n4. Testing Search Endpoint (POST request):"
curl -s -X POST http://localhost:8081/search \
  -H "Content-Type: application/json" \
  -d '{"query": "love your enemies", "k": 3}' | jq .

# Test search with GET filters
echo -e "\n5. Testing Search with GET filters:"
curl -s "http://localhost:8081/search?q=blessed&book=Matthew&chapter=5&k=3" | jq .

# Test search with inline filters
echo -e "\n6. Testing Search with inline filters (POST):"
curl -s -X POST http://localhost:8081/search \
  -H "Content-Type: application/json" \
  -d '{"query": "blessed book:Psalms", "k": 3}' | jq .

echo -e "\nAll tests complete!"