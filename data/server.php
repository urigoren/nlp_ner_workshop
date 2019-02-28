<?php
/*
    Code for submission server, live at:
    http://www.goren4u.com/nlp_ner
*/
function cmp_lines($a, $b) {
    $lcmp = array_combine($a, $b); // zip(a, b)
    $ret = 0;
    $n = 0;
    foreach($lcmp as $x => $y) {
        $n+=1;
        $ret+=(trim($x) == trim($y)) ? 1 : 0;
    }
    return $ret/$n;
}

if (array_key_exists("submission", $_REQUEST) && array_key_exists("name", $_REQUEST)) //Record submission
{
        $user=preg_replace("/[^a-zA-Z0-9_]+/", "", $_REQUEST["name"]);
        $submission=json_decode($_REQUEST["submission"], true);
        $truth=json_decode(file_get_contents("holdout.data"), true);
        // Calculate the accuracy of the submission
        $n=0;$score=0;
        foreach($truth as $x => $y) {
            $n +=1;
            $score += array_key_exists($x, $submission) ? cmp_lines($submission[$x], $y) : 0;
        }
        $score /= $n;
        $submission["score"] = $score;
        $submission["user"] = $user;
        // Save submission as json file
        file_put_contents("$user.json",json_encode($submission));
        echo $score;
}
else // Show leaderboard
{
        // flat files to "user"=>"score" array
        $scores = array();
        $files = scandir('.');
        foreach ($files as $index=>$fname) {
                if (substr($fname,-5)=='.json') {
                        $json= json_decode(file_get_contents($fname), true);
                        $scores[$json["user"]] = $json["score"];
                }
        }
        arsort($scores);
        // Format the leaderboard
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <title>NLP Named-Entity-Recognition Workshop</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.6/umd/popper.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.2.1/js/bootstrap.min.js"></script>
  <meta http-equiv="refresh" content="30" />
</head>
<body><div class="container" align="center">
<h1>Leader board</h1>
  <table class="table table-striped">
    <thead>
      <tr>
        <th>Submission_id</th>
        <th>Accuracy</th>
      </tr>
    </thead>
    <tbody>
<?php
        foreach($scores as $user => $score) {
                echo "<tr><th>$user</th><td>$score</td></tr>";
        }
?>
</tbody></table>
<br />
Made by <a href="http://www.goren4u.com">Uri Goren</a>.
</div></body></html>
<?php
}
?>

