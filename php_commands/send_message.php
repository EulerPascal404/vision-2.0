<?php

// Facebook Graph API URL
$url = 'https://graph.facebook.com/v20.0/445630435301408/messages';

// The access token (replace this with your actual token)
$accessToken = 'EAAMHRIDP1DYBO107OIybJMiyYnxCFlUWHkDChf3YknKAE37lM2FY2ZCPIfhUtHzxhbMkJAHx7poZAfCZBwlLTQ1iEHJKpI4hI1eLVaVTZCkJZBQiybYDaZCFDKkNiAL55WYCczLLjR3VJ7Wzo56HMZBDJZAb8YBo9ruVZCvZAmZArcZB7NROIaSAFppYlheTAJZCbfEK69zdI1g3xpAJSOqtRSW1mJGv9LvwZD';

// WhatsApp message payload
$data = [
    'messaging_product' => 'whatsapp',
    'to' => '14083451612',  // Replace with the recipient's phone number
    'type' => 'template',
    'template' => [
        'name' => 'hello_world',
        'language' => [
            'code' => 'en_US'
        ]
    ]
];

// Convert data to JSON
$jsonData = json_encode($data);

// Initialize cURL session
$ch = curl_init();

// Set cURL options using curl_setopt_array
curl_setopt_array($ch, [
    CURLOPT_URL => $url,                              // URL to send the request to
    CURLOPT_RETURNTRANSFER => true,                   // Return the response instead of outputting it
    CURLOPT_POST => true,                             // Use POST method
    CURLOPT_HTTPHEADER => [                           // Set headers
        'Authorization: Bearer ' . $accessToken,      // Bearer token for authorization
        'Content-Type: application/json'              // Content type as JSON
    ],
    CURLOPT_POSTFIELDS => $jsonData,                  // The JSON payload to send
    CURLOPT_HEADER => true                            // Include headers in the output
]);

// Execute the cURL request and get the response
$response = curl_exec($ch);

// Check for cURL errors
if (curl_errno($ch)) {
    echo 'cURL error: ' . curl_error($ch);
} else {
    // Output the response from the API
    echo 'Response: ' . $response;
}

// Close the cURL session
curl_close($ch);

?>

