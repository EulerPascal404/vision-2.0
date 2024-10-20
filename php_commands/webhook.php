<?php 
require 'vendor/autoload.php';

function getGPTResponse($query)
{
    // python needs to implement an endpoint: ENDPOINT
    // php sends HTTP POST request to ENDPOINT
    // python returns response
    // php receives response and returns response
    $url = 'http://127.0.0.1:5000/gemini_endpoint';
    $data = [
        'name' => $query,
        'age' => 30
    ];
    $ch = curl_init($url);

// Set options for the cURL transfer
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true); // Return the response as a string
    curl_setopt($ch, CURLOPT_POST, true); // Specify that this is a POST request
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data)); // Set the POST fields (data) as JSON
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']); // Set content type to JSON

    // Execute the cURL request
    $response = curl_exec($ch);

    curl_close($ch);
/*
    $yourApiKey = getenv(OPENAI_KEY);
    $client = Gemini::client($yourApiKey);
    //$model = $client->geminiPro()->generativeModel(model: 'models/gemini-1.5-flash-001');
    $model = $client->geminiPro();
    $result = $model->generateContent("$query");
    /*
    $data = $client->chat()->create([
        'model' => 'gpt-3.5-turbo',
        'messages' => [[
            'role' => 'user',
            'content' => "Limit your response to 100 characters for this query: $query",
        ]],
    ]);*/
    return $result->text();

    return $data['choices'][0]['message']['content'];
}

function getGPTImageResponse($query = "What’s in this image? ")
{
    $base64Image = encodeImage(SAVE_IMAGE_PATH);
    $url = 'http://127.0.0.1:5000/gemini_endpoint';
    $data = [
        'name' => $query,
        'age' => $base64Image
    ];

    $ch = curl_init($url);

// Set options for the cURL transfer
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true); // Return the response as a string
    curl_setopt($ch, CURLOPT_POST, true); // Specify that this is a POST request
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data)); // Set the POST fields (data) as JSON
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']); // Set content type to JSON

    // Execute the cURL request
    $response = curl_exec($ch);

    curl_close($ch);
    /*$client = OpenAI::client(OPENAI_KEY);
    $data = $client->chat()->create([
        'model'      => 'gpt-4-vision-preview',
        'messages'   => [
            [
                'role'    => 'user',
                'content' => [
                    [
                        'type' => 'text',
                        'text' => "$query, limit your response to 100 characters or less."
                    ],
                    [
                        'type' => 'image_url',
                        'image_url' => "data:image/jpeg;base64,$base64Image"
                    ],
                ],
            ]
        ],
        'max_tokens' => 200,
    ]);

    return $data['choices'][0]['message']['content'];*/
    return 0;
}

function sendWhatsappResponse($response)
{
    $curl = curl_init();
    curl_setopt_array($curl, array(
        CURLOPT_URL => 'https://graph.facebook.com/v20.0/' . WHATSAPP_SENDER_ID . '/messages',
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_ENCODING => '',
        CURLOPT_MAXREDIRS => 10,
        CURLOPT_TIMEOUT => 0,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
        CURLOPT_CUSTOMREQUEST => 'POST',
        CURLOPT_POSTFIELDS => '{"messaging_product": "whatsapp", "to": "' . WHATSAPP_INCOMING_PHONE_NUMBER . '","text": {"body" : "' . $response . '"}}',
        CURLOPT_HTTPHEADER => array(
            'Authorization: Bearer ' . WHATSAPP_TOKEN,
            'Content-Type: application/json'
        ),
    ));

    $response = curl_exec($curl);
    curl_close($curl);
}
function getMediaLink($mediaID)
{
    $curl = curl_init();
    curl_setopt_array($curl, array(
        CURLOPT_URL => 'https://graph.facebook.com/v20.0/' . $mediaID,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_ENCODING => '',
        CURLOPT_MAXREDIRS => 10,
        CURLOPT_TIMEOUT => 0,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
        CURLOPT_CUSTOMREQUEST => 'GET',
        CURLOPT_HTTPHEADER => array(
            'Authorization: Bearer ' . WHATSAPP_TOKEN,
        ),
    ));

    $response = curl_exec($curl);
    curl_close($curl);

    $myfile = fopen("newfile.txt", "w");
    fwrite($myfile, $response);
    fclose($myfile);
    return json_decode($response)->url;
}

function downloadMediaLink($url)
{
    $curl = curl_init();
    $fp = fopen(SAVE_IMAGE_PATH, 'w+');
    curl_setopt_array($curl, array(
        CURLOPT_URL => $url,
        CURLOPT_FILE => $fp,
        CURLOPT_ENCODING => '',
        CURLOPT_MAXREDIRS => 10,
        CURLOPT_USERAGENT => 'PostmanRuntime/7.36.0',
        CURLOPT_TIMEOUT => 0,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_HTTP_VERSION => CURL_HTTP_VERSION_1_1,
        CURLOPT_CUSTOMREQUEST => 'GET',
        CURLOPT_HTTPHEADER => array(
            'Authorization: Bearer ' . WHATSAPP_TOKEN
        ),
    ));

    curl_exec($curl);
    curl_close($curl);
    fclose($fp);
}

function encodeImage($imagePath): string
{
    $imageContent = file_get_contents($imagePath);
    return base64_encode($imageContent);
}

// CHANGE THESE!
define("WHATSAPP_TOKEN", "EAAMHRIDP1DYBOyJGDSQ0PmzIO0XrEfj0brQpqomCl6pIh8Uv3RzoZBsKGCvcrq4IONDhhEDQWGFb3goot0ZAseZCNLZAMNJESVfKo5z3qxg8tywPKUn91HqrBjOR8TZBS5qIdKuEaFugpFdxVFPDTkCDQjfvryXf2yq8EN8vGaqooxGFrw9QvyYF6CVY5PEbEMoybppmIFoIc8Lm4BlCequAtbyUZD");
define("WHATSAPP_SENDER_ID", "445630435301408");
define("WHATSAPP_INCOMING_PHONE_NUMBER", "14083451612");
define("OPENAI_KEY", "AIzaSyA9Ddx0pD-qN9rzbCVaMQ1eNAW5BeIXTqw");


define("SAVE_IMAGE_PATH", "query_image.jpg");


if (isset($_GET['hub_challenge'])) {
    // Used for verification of Whatsapp
    echo ($_GET['hub_challenge']);
} else {
    $json = file_get_contents('php://input');

    // Uncomment the following lines if you need to read the incoming data from the Whastapp webhook
    // file_put_contents("debug.txt", $json);
    // die();

    $json = json_decode($json);

    $message = $json->entry[0]->changes[0]->value->messages[0];

    echo "HELLO";
    $url = 'http://127.0.0.1:5000/gemini_endpoint';
    echo $message->from;

    if ($message->from == WHATSAPP_INCOMING_PHONE_NUMBER) {
        echo "HELLO WORLD";
        if ($message->type == "text") {
            /*$query = $message->text->body;
            if (file_exists(SAVE_IMAGE_PATH)) {
                $response = getGPTImageResponse($query);
                unlink(SAVE_IMAGE_PATH);
            } else {
                $response = getGPTResponse($query);
            }
            sendWhatsappResponse($response);*/
        } else if ($message->type == "image") {
            $url = 'http://127.0.0.1:5000/gemini_endpoint';
            $mediaLink = getMediaLink($message->image->id);
            downloadMediaLink($mediaLink);
            getGPTImageResponse();
        }
    }
}
/*
$verify_token = "HAPPY";


if (isset($_GET['hub_mode']) && $_GET['hub_mode'] === 'subscribe' &&
    $_GET['hub_verify_token'] === $verify_token) {
    // Return the hub_challenge to verify
    echo $_GET['hub_challenge'];
    exit;
}

$payload = file_get_contents('php://input');
file_put_contents('webhook.log', date('Y-m-d H:i:s') . " - " . $payload . "\n", FILE_APPEND); 
http_response_code(200); 
echo 'HAPPY';*/
?>
