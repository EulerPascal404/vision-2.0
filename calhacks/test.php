$curlCommand = "curl -X GET 'https://api.example.com/data' -H 'Authorization: Bearer your_token_here'"; // replace the link and token
$output = shell_exec($curlCommand);