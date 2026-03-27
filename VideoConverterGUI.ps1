# ================================================
# Video Converter GUI (PowerShell + WPF)
# Calls your ffmpeg batch logic with selected files
# ================================================

Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName System.Windows.Forms

# ----------------- XAML for the GUI -----------------
[xml]$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="Video Converter - HEVC NVENC" Height="600" Width="900" Background="#2D2D2D" Foreground="White">
    <Grid Margin="10">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>

        <!-- Header -->
        <StackPanel Grid.Row="0" Orientation="Horizontal" Margin="0,0,0,10">
            <TextBlock Text="Video Converter (HEVC NVENC)" FontSize="18" FontWeight="Bold" VerticalAlignment="Center"/>
            <TextBlock x:Name="StatusText" Margin="20,0,0,0" VerticalAlignment="Center" Foreground="#00FF00"/>
        </StackPanel>

        <!-- Controls -->
        <StackPanel Grid.Row="1" Orientation="Horizontal" Margin="0,0,0,10">
            <Button x:Name="btnAddFiles" Content="Add Video Files" Width="140" Height="30" Margin="0,0,10,0"/>
            <Button x:Name="btnAddFolder" Content="Add Folder (Recursive)" Width="160" Height="30" Margin="0,0,10,0"/>
            <Button x:Name="btnClear" Content="Clear List" Width="100" Height="30" Margin="0,0,10,0"/>
            
            <TextBlock Text="CQ Quality:" Margin="20,5,5,0" VerticalAlignment="Center"/>
            <TextBox x:Name="txtCQ" Text="28" Width="60" Height="30" Margin="0,0,10,0" HorizontalContentAlignment="Center"/>
            
            <TextBlock Text="Output Base Folder:" Margin="20,5,5,0" VerticalAlignment="Center"/>
            <TextBox x:Name="txtOutputBase" Width="280" Height="30" Margin="0,0,10,0" Text="E:\Photos\tmp_videos"/>
            <Button x:Name="btnBrowseOutput" Content="..." Width="30" Height="30"/>
        </StackPanel>

        <!-- File List -->
        <ListView x:Name="lvFiles" Grid.Row="2" Margin="0,0,0,10">
            <ListView.View>
                <GridView>
                    <GridViewColumn Header="File Path" Width="700" DisplayMemberBinding="{Binding FullPath}"/>
                    <GridViewColumn Header="Size" Width="100" DisplayMemberBinding="{Binding SizeMB}"/>
                </GridView>
            </ListView.View>
        </ListView>

        <!-- Bottom Buttons -->
        <StackPanel Grid.Row="3" Orientation="Horizontal" HorizontalAlignment="Right">
            <Button x:Name="btnConvert" Content="Start Conversion" Width="160" Height="40" Background="#007ACC" Foreground="White" FontWeight="Bold" Margin="0,0,10,0"/>
            <Button x:Name="btnCancel" Content="Cancel" Width="100" Height="40" Background="#555555" Foreground="White"/>
        </StackPanel>
    </Grid>
</Window>
"@

# Load XAML
$reader = New-Object System.Xml.XmlNodeReader $xaml
$window = [Windows.Markup.XamlReader]::Load($reader)

# Get controls
$btnAddFiles = $window.FindName("btnAddFiles")
$btnAddFolder = $window.FindName("btnAddFolder")
$btnClear = $window.FindName("btnClear")
$btnBrowseOutput = $window.FindName("btnBrowseOutput")
$btnConvert = $window.FindName("btnConvert")
$btnCancel = $window.FindName("btnCancel")

$lvFiles = $window.FindName("lvFiles")
$txtCQ = $window.FindName("txtCQ")
$txtOutputBase = $window.FindName("txtOutputBase")
$statusText = $window.FindName("StatusText")

# Data for ListView
$videoList = New-Object System.Collections.ObjectModel.ObservableCollection[object]

# Function to add files
function Add-VideoFiles {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Multiselect = $true
    $dialog.Filter = "Video Files|*.mp4;*.mkv;*.mov;*.avi;*.m4v;*.ts;*.m2ts|All Files|*.*"
    
    if ($dialog.ShowDialog() -eq "OK") {
        foreach ($file in $dialog.FileNames) {
            $item = [PSCustomObject]@{
                FullPath = $file
                SizeMB   = "{0:N1} MB" -f ((Get-Item $file).Length / 1MB)
            }
            $videoList.Add($item)
        }
    }
}

# Function to add folder recursively
function Add-VideoFolder {
    $folderDialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderDialog.Description = "Select folder containing videos (will search recursively)"
    
    if ($folderDialog.ShowDialog() -eq "OK") {
        $files = Get-ChildItem -Path $folderDialog.SelectedPath -Recurse -Include *.mp4,*.mkv,*.mov,*.avi,*.m4v,*.ts,*.m2ts -File
        foreach ($file in $files) {
            $item = [PSCustomObject]@{
                FullPath = $file.FullName
                SizeMB   = "{0:N1} MB" -f ($file.Length / 1MB)
            }
            $videoList.Add($item)
        }
    }
}

# Button events
$btnAddFiles.Add_Click({ Add-VideoFiles })
$btnAddFolder.Add_Click({ Add-VideoFolder })
$btnClear.Add_Click({ $videoList.Clear() })

$btnBrowseOutput.Add_Click({
    $folderDialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $folderDialog.SelectedPath = $txtOutputBase.Text
    if ($folderDialog.ShowDialog() -eq "OK") {
        $txtOutputBase.Text = $folderDialog.SelectedPath
    }
})

# Main Conversion Function
$btnConvert.Add_Click({
    if ($videoList.Count -eq 0) {
        [System.Windows.MessageBox]::Show("Please add at least one video file.", "Warning", "OK", "Warning")
        return
    }

    $cq = $txtCQ.Text
    $outputBase = $txtOutputBase.Text.TrimEnd('\')

    if (-not (Test-Path $outputBase)) {
        try { New-Item -Path $outputBase -ItemType Directory -Force | Out-Null }
        catch {
            [System.Windows.MessageBox]::Show("Cannot create output folder.", "Error", "OK", "Error")
            return
        }
    }

    $statusText.Text = "Starting conversion of $($videoList.Count) videos..."
    $btnConvert.IsEnabled = $false

    # Create videos.txt for the batch script
    $tempTxt = "$env:TEMP\videos_$(Get-Random).txt"
    $videoList.FullPath | Out-File -FilePath $tempTxt -Encoding utf8

    # Build the command similar to your .bat but directly in PowerShell for better control
    $scriptBlock = {
        param($videosTxt, $outputBase, $cq)

        foreach ($videoPath in Get-Content $videosTxt) {
            if (-not (Test-Path $videoPath)) { continue }

            $file = Get-Item $videoPath
            $parentName = $file.Directory.Name
            $outputDir = Join-Path $outputBase $parentName
            New-Item -Path $outputDir -ItemType Directory -Force | Out-Null

            $outputFile = Join-Path $outputDir "$($file.BaseName).mkv"

            Write-Host "Processing: $($file.Name)" -ForegroundColor Cyan

            & ffmpeg -i "$videoPath" `
                -map 0:v:0 `
                -map 0:a `
                -map 0:2 `
                -vf "scale=-2:720" `
                -c:v hevc_nvenc `
                -rc vbr `
                -cq $cq `
                -b:v 0 `
                -preset p5 `
                -tier high `
                -pix_fmt yuv420p `
                -c:a copy `
                -c:s copy `
                "$outputFile" 2>&1 | ForEach-Object { Write-Host $_ }
        }

        Remove-Item $videosTxt -Force -ErrorAction SilentlyContinue
    }

    # Run conversion in background job so GUI doesn't freeze
    $job = Start-Job -ScriptBlock $scriptBlock -ArgumentList $tempTxt, $outputBase, $cq

    # Monitor job
    while ($job.State -eq "Running") {
        Start-Sleep -Milliseconds 500
        $statusText.Text = "Converting... $($videoList.Count) files in progress"
        [System.Windows.Threading.Dispatcher]::CurrentDispatcher.Invoke([Action]{}, [Windows.Threading.DispatcherPriority]::Background)
    }

    $statusText.Text = "Conversion finished!"
    $btnConvert.IsEnabled = $true

    [System.Windows.MessageBox]::Show("Conversion completed!", "Success", "OK", "Information")
})

$btnCancel.Add_Click({ $window.Close() })

# Set ListView data source
$lvFiles.ItemsSource = $videoList

# Show the window
$window.ShowDialog() | Out-Null