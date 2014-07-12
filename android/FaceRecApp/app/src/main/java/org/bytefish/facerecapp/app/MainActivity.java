package org.bytefish.facerecapp.app;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

public class MainActivity extends ActionBarActivity {

    private static final int CAPTURE_IMAGE_REQUEST_CODE = 100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle presses on the action bar items
        switch (item.getItemId()) {
            case android.R.id.home:
                // app icon in action bar clicked; go home
                Intent intentHome = new Intent(this, MainActivity.class);
                intentHome.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivity(intentHome);
                return true;
            case R.id.action_search:
                // TODO
                Toast.makeText(getApplicationContext(), "Not implemented!", Toast.LENGTH_SHORT)
                        .show();
                return true;
            case R.id.action_capture:
                runCaptureImageActivity();
                return true;
            case R.id.action_picture:
                runSelectImageActivity();
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    private void runSelectImageActivity() {
        startActivity(new Intent(MainActivity.this, SelectImageActivity.class));
    }

    private void runCaptureImageActivity() {
        startActivity(new Intent(MainActivity.this, CaptureActivity.class));
    }

}
